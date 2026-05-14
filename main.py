import os
import shutil
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim

from common.opt import opts
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='./runs/' + opt.model_name)


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split == 'train':
            output_3D = model(input_2D)
        else:
            input_2D, output_3D = input_augmentation(input_2D, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
        loss = weighted_mpjpe(output_3D, out_target, w_mpjpe)

        N = input_2D.size(0)
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D_eval = output_3D[:, opt.pad].unsqueeze(1)
            output_3D_eval[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D_eval, out_target, action, action_error_sum, opt.dataset,
                                                subject)

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return loss_all['loss'].avg, p1, p2


def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]
    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip = model(input_2D_flip)
    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip
    return input_2D, output_3D


if __name__ == '__main__':
    manualSeed = opt.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("lr: ", opt.lr)
    print("batch_size: ", opt.batch_size)
    print("channel: ", opt.channel)
    print("GPU: ", opt.gpu)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    checkpoint_dir = os.path.join('ckpt', opt.model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_filename = getattr(opt, 'model', 'dc_gct_v1')

    try:
        model_src_path = os.path.join("model", f"{model_filename}.py")
        model_dst_path = os.path.join(checkpoint_dir, f"{model_filename}.py")
        shutil.copy(model_src_path, model_dst_path)

        main_src_path = "main.py"
        main_dst_path = os.path.join(checkpoint_dir, "main.py")
        shutil.copy(main_src_path, main_dst_path)

        block_src_dir = os.path.join("model", "block")
        block_dst_dir = os.path.join(checkpoint_dir, "block")
        if os.path.exists(block_src_dir):
            shutil.copytree(block_src_dir, block_dst_dir, dirs_exist_ok=True)
    except Exception as e:
        print("备份文件时发生错误, 继续运行...", e)

    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    class_name = "DC_GCT"
    exec(f'from model.{model_filename} import {class_name}', globals())
    model = eval(f'{class_name}(opt).cuda()')

    if opt.reload:
        model_dict = model.state_dict()
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))[0]
        print("Reloading from:", model_path)
        pre_dict = torch.load(model_path)
        for name, key in model_dict.items():
            if name in pre_dict:
                model_dict[name] = pre_dict[name]
        model.load_state_dict(model_dict)

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000)

    all_param = list(model.parameters())
    lr = opt.lr
    optimizer = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    best_epoch = 0

    for epoch in range(1, opt.nepoch):
        train_loss = 0.0
        if opt.train:
            train_loss = train(opt, actions, train_dataloader, model, optimizer, epoch)

        val_loss, p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train:
            writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('loss/val_loss', val_loss, epoch)
        writer.add_scalar('mpjpe', p1, epoch)
        writer.add_scalar('p2', p2, epoch)

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
            opt.previous_best_threshold = p1
            best_epoch = epoch

        if opt.train == 0:
            print('val_loss: %.4f, p1: %.2f, p2: %.2f' % (val_loss, p1, p2))
            break
        else:
            log_str = 'epoch: %d, lr: %.7f, train_loss: %.4f, val_loss: %.4f, p1: %.2f, p2: %.2f | Best Epoch: %d, Best p1: %.2f' % (
                epoch, lr, train_loss, val_loss, p1, p2, best_epoch, opt.previous_best_threshold)
            logging.info(log_str)
            print(log_str.replace('epoch:', 'e:'))

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay