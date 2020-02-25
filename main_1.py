import os
import json
import socket
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

from opts import parse_opts
from dataloaders.dataset import get_training_set, get_validation_set

from model import generate_model
from utils import Logger, get_fine_tuning_parameters
from train_1 import train_epoch
from validation import val_epoch

import torchvision.transforms as transforms
from temporal_transforms import TemporalSegmentRandomCrop, TemporalSegmentCenterCrop
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from warmup_scheduler import GradualWarmupScheduler
from radam import RAdam
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':

    # set parameters
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    opt = parse_opts()
    torch.manual_seed(opt.manual_seed)
    opt.timestamp = timestamp
    if opt.dataset == 'CCV':
        opt.event_classes = 21
    elif opt.dataset == 'FCVID':
        opt.event_classes = 238
    elif opt.dataset == 'ActivityNet':
        opt.event_classes = 200
    elif opt.dataset == 'EPIC':
        opt.event_classes = (125, 352)

    # set path
    if opt.data_root_path != '':
        opt.video_path = os.path.join(
            opt.data_root_path, opt.dataset, opt.video_path)
        if opt.dataset == 'EPIC':
            opt.annotation_path = os.path.join(
                opt.data_root_path,
                opt.dataset,
                opt.annotation_path)
        else:
            opt.annotation_path = os.path.join(
                opt.data_root_path,
                opt.dataset,
                opt.annotation_path +
                opt.dataset +
                '.json')
    if opt.result_path != '':
        opt.result_path = os.path.join(
            opt.result_path,
            opt.dataset,
            opt.model_name)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
        opt.save_path = os.path.join(opt.result_path, timestamp)
        os.makedirs(opt.save_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    # save parameters
    with open(os.path.join(opt.save_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    print(opt)

    # Logging Tensorboard
    log_dir = os.path.join(
        opt.save_path,
        'tensorboard',
        'runs',
        socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='-params')

    # model and criterion
    model, parameters = generate_model(opt)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelSoftMarginLoss()

    # set cuda
    if not opt.no_cuda:
       os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
       torch.cuda.manual_seed(opt.manual_seed)

    opt.device = torch.device("cuda" if not opt.no_cuda else "cpu")
    criterion = criterion.to(opt.device)
    model = model.to(opt.device)

    #model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if opt.ngpus > 1:
        model = torch.nn.DataParallel(model)
    #   model = encoding.parallel.DataParallelModel(model, device_ids=[6,7])
    #   criterion = encoding.parallel.DataParallelCriterion(criterion, device_ids=[6,7])

    ## optimizer
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=opt.learning_rate,
            betas=(0.9, 0.999),
            # eps=1e-8,
            eps=3e-4,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adadelta':
        optimizer = optim.Adadelta(
            parameters,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'radam':
        optimizer = RAdam(
            parameters,
            lr=opt.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=opt.weight_decay)

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ## prepare train
    if not opt.no_train:
        temporal_transform = TemporalSegmentRandomCrop(opt.segment_number, opt.sample_duration)

        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            sceobj_crop_method = MultiScaleRandomCrop(opt.scales, opt.sceobj_frame_size)
        elif opt.train_crop == 'corner':
            sceobj_crop_method = MultiScaleCornerCrop(opt.scales, opt.sceobj_frame_size)
        elif opt.train_crop == 'center':
            sceobj_crop_method = MultiScaleCornerCrop(opt.scales, opt.sceobj_frame_size, crop_positions=['c'])
        sceobj_spatial_transform = Compose([
            sceobj_crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value),
            normalize
        ])
        #sceobj_spatial_transform = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(opt.sceobj_frame_size),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #])
        training_data = get_training_set(opt, sceobj_spatial_transform, temporal_transform)
        print('len of training data', len(training_data))

        train_loader = DataLoaderX(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)
        train_logger = Logger(
            os.path.join(opt.save_path, 'train.log'),
            # ['epoch', 'loss', 'final_mAP_sigmoid', 'final_mAP_softmax', 'lr', 'verb_top1', 'verb_top5']
            ['epoch', 'top1', 'top5', 'verb_top1', 'verb_top5', 'noun_top1', 'noun_top5',
             'loss', 'verb_loss', 'noun_loss', 'lr'])

    ## prepare validation
    if not opt.no_val:
        temporal_transform = TemporalSegmentCenterCrop(opt.segment_number, opt.sample_duration)
        sceobj_spatial_transform = Compose([
            Scale(opt.sceobj_frame_size),
            CenterCrop(opt.sceobj_frame_size),
            ToTensor(opt.norm_value),
            normalize
        ])
        #sceobj_spatial_transform = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(opt.sceobj_frame_size),
        #    #transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #])
        validation_data = get_validation_set(opt, sceobj_spatial_transform, temporal_transform)

        val_loader = DataLoaderX(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)

        val_logger = Logger(
            os.path.join(opt.save_path, 'val.log'),
			# ['epoch', 'final_mAP_sigmoid', 'final_mAP_softmax', 'verb_top1', 'verb_top5']
            ['epoch', 'top1', 'top5', 'verb_top1', 'verb_top5', 'noun_top1', 'noun_top5',
             'loss', 'verb_loss', 'noun_loss'])

    ## train process
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location=opt.device)

        opt.begin_epoch = checkpoint['epoch']
        #state_dict = {str.replace(k, 'module.', ''):v for k, v in checkpoint['state_dict'].items()}
        #model.load_state_dict(state_dict)
        model.load_state_dict(checkpoint['state_dict'])
        #if not opt.no_train:
        #    optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint

    # train and validation

    #val_epoch(opt.begin_epoch, val_loader, model, opt, val_logger, writer)

    ## scheduler one
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.lr_decay)

    ## scheduler two
    #scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, opt.n_epochs)
    #scheduler_cosine = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.lr_decay)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=4, after_scheduler=scheduler_cosine)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=1000, total_epoch=6, after_scheduler=scheduler_cosine)

    stats_dict = dict(train_loss=np.zeros((opt.n_epochs+1,)),
                      train_verb_loss=np.zeros((opt.n_epochs+1,)),
                      train_noun_loss=np.zeros((opt.n_epochs+1,)),
                      train_acc=np.zeros((opt.n_epochs+1,)),
                      train_verb_acc=np.zeros((opt.n_epochs+1,)),
                      train_noun_acc=np.zeros((opt.n_epochs+1,)),
                      val_loss=np.zeros((opt.n_epochs+1,)),
                      val_verb_loss=np.zeros((opt.n_epochs+1,)),
                      val_noun_loss=np.zeros((opt.n_epochs+1,)),
                      val_acc=np.zeros((opt.n_epochs+1,)),
                      val_verb_acc=np.zeros((opt.n_epochs+1,)),
                      val_noun_acc=np.zeros((opt.n_epochs+1,))
    )

    for _ in range(1, opt.begin_epoch):
        scheduler.step()

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            cudnn.benchmark = True
            training_metrics = train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, writer)
            for k, v in training_metrics.items():
                stats_dict[k][i] = v

        if i % opt.checkpoint == 0:
            save_file_path = os.path.join(opt.save_path, 'train_' + str(i) + '_model.pth')
            states = {
                'epoch': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),}
            torch.save(states, save_file_path)

        if not opt.no_val and i % opt.val_per_epoches == 0:
            test_metrics = val_epoch(i, val_loader, model, criterion, opt, val_logger, writer)
            for k, v in test_metrics.items():
                stats_dict[k][i] = v

        scheduler.step()

    writer.close()

    save_stats_dir = os.path.join(opt.save_path, 'stats')
    if not os.path.exists(save_stats_dir):
        os.makedirs(save_stats_dir)
    with open(os.path.join(save_stats_dir, 'training_stats.npz'), 'wb') as f:
        np.savez(f, **stats_dict)
