import os
from dataloaders.fcvid import FCVID
from dataloaders.ccv import CCV
from dataloaders.activitynet import ActivityNet
from dataloaders.epic import EPIC


def get_training_set(opt, sceobj_spatial_transform, temporal_transform):

    if opt.dataset == 'FCVID':
        training_data = FCVID(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'CCV':
        training_data = CCV(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'ActivityNet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'EPIC':
        training_data = EPIC(
            opt.data_root_path,
            opt.video_path,
            os.path.join(opt.annotation_path, 'EPIC_train_action_labels.pkl'),
            class_type='verb',
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform)
    return training_data


def get_validation_set(opt, sceobj_spatial_transform, temporal_transform):

    if opt.dataset == 'FCVID':
        validation_data = FCVID(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'CCV':
        validation_data = CCV(
                opt.video_path,
                opt.annotation_path,
                'validation',
                spatial_transform=sceobj_spatial_transform,
                temporal_transform=temporal_transform)
    elif opt.dataset == 'ActivityNet':
        validation_data = ActivityNet(
                opt.video_path,
                opt.annotation_path,
                'validation',
                spatial_transform=sceobj_spatial_transform,
                temporal_transform=temporal_transform)
    elif opt.dataset == 'EPIC':
        validation_data = EPIC(
            opt.data_root_path,
            opt.video_path,
            os.path.join(opt.annotation_path, 'EPIC_val_action_labels.pkl'),
            class_type='verb',
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform)
    return validation_data
