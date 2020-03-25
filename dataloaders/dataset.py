import os
from dataloaders.fcvid import FCVID
from dataloaders.ccv import CCV
from dataloaders.activitynet import ActivityNet
from dataloaders.epic import EPIC
from epic_kitchens.meta import test_timestamps


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
            opt.video_path,
            os.path.join(opt.annotation_path, 'EPIC_train_action_labels.pkl'),
            mode='train',
            obj_feature_type=opt.object_feature_type,
            dataset_break=opt.dataset_break,
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform,
            object_feature_path=opt.object_feature_path,
            use_obj_feature=opt.use_object_feature)
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
            opt.video_path,
            os.path.join(opt.annotation_path, 'EPIC_val_action_labels.pkl'),
            mode='val',
            object_feature_path=opt.object_feature_path,
            dataset_break=opt.dataset_break,
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform,
            obj_feature_type=opt.object_feature_type,
            use_obj_feature=opt.use_object_feature)
    return validation_data


def get_test_set(opt, sceobj_spatial_transform, temporal_transform, test_set='seen'):
    video_path = opt.video_path.split('/')[-1]

    if test_set == 'seen':
        test_data = EPIC(
            opt.video_path.replace(video_path, 'test_seen_' + video_path),
            os.path.join(opt.annotation_path, 'EPIC_test_s1_timestamps.pkl'),
            mode='test',
            dataset_break=opt.dataset_break,
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform,
            object_feature_path=opt.object_feature_path,
            obj_feature_type=opt.object_feature_type,
            use_obj_feature=opt.use_object_feature)
    elif test_set == 'unseen':
        test_data = EPIC(
            opt.video_path.replace(video_path, 'test_unseen_' + video_path),
            os.path.join(opt.annotation_path, 'EPIC_test_s2_timestamps.pkl'),
            mode='test',
            dataset_break=opt.dataset_break,
            spatial_transform=sceobj_spatial_transform,
            temporal_transform=temporal_transform,
            object_feature_path=opt.object_feature_path,
            obj_feature_type=opt.object_feature_type,
            use_obj_feature=opt.use_object_feature)
    return test_data