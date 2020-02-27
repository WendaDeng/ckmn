import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    # model explain
    parser.add_argument(
        '--model_name',
        #default='FtDetectorFc-Sc',
        #default='FtDetectorFc-Ob',
        #default='FtDetectorFc-Ac',
        #default='FtDetectorFc-Scob',
        default='FtDetectorFc-3detectors',
        #default='FtDetectorFc-3detectors-Nonlocal',
        #default='FtDetectorFc-3detectors-Dilated',
        #default='FtDetectorFc-3detectors-Dilated-Nonlocal',
        #default='FtDetectorFc-3detectors-Nonlocal-Dilated',
        #default='FtDetectorFc-3detectors-Coatt',
        #default='FtDetectorFc-3detectors-Coatt-Trilinear',
        #default='FtDetectorFc-3detectors-Coatt-Bilinear',
        #default='FtDetectorFc-3detectors-Coatt-Bitrilinear',
        #default='FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Serial-Concat',
        #default='FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Serial-Trilinear',
        # default='FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Parallel-Concat',
        #default='FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Parallel-Bilinear',
        #default='FtDetectorFc-3detectors-DilatedNonlocal-Coatt-General-Parallel-Trilinear',
        #default='FtDetectorFc-3detectors-Nonlocal-Coatt-Bitrilinear',
        type=str,
        help='')

    # datasets
    parser.add_argument(
        '--dataset',
        default='EPIC',
        # default='FCVID',
        #default='CCV',
        #default='ActivityNet',
        type=str,
        help='Used dataset (FCVID | CCV | ActivityNet | EPIC)')

    # path
    parser.add_argument(
        '--data_root_path',
        # default='/DATA/disk1/qzb/datasets/',
        default='/mnt/shared_40t/dwd/epic-kitchens/orn/data/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        # default='frames',
        default='videos_256x256_30',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        # default='video_labels/',
        default='meta',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='./results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--resume_path',
        default='',
        # default='resume/train_1_model.pth',
        type=str,
        help='Save data (.pth) of previous training')

    # models
    parser.add_argument(
        '--scene_base_model',
        default='resnet50',
        type=str,
        help='concept detector base model (resnet50)')
    parser.add_argument(
        '--object_base_model',
        default='resnet50',
        type=str,
        help='concept detector base model (resnet50 | resnet101 | fasterrcnn_resnet50_fpn)')
    parser.add_argument(
        '--action_base_model',
        default='I3D',
        type=str,
        help='concept detector base model (resnet50 | resnet101 | resnext101 | I3D)')
    parser.add_argument(
        '--general_base_model',
        # default='resnext101',
        default='I3D',
        type=str,
        help='general feature model type (resnet50 | resnet101 | resnext101 | I3D)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--scene_classes',
        default=365,
        type=int,
        help='Number of scene classes')
    parser.add_argument(
        '--object_classes',
        default=1000,
        type=int,
        help='Number of object classes')
    parser.add_argument(
        '--action_classes',
        default=400,
        type=int,
        help='Number of action classes')

    # gpu
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_ids',
        default='0',
        type=str,
        help='GPU ID')
    parser.add_argument(
        '--ngpus',
        default=1,
        type=int,
        help='GPU numbers')

    # data process
    parser.add_argument(
        '--val_batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument(
        '--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument(
        '--segment_number',
        default=8,
        type=int,
        help='Number of video segments')
    parser.add_argument(
        '--sample_duration',
        default=8,
        type=int,
        help='Temporal duration of each video segment')
    parser.add_argument(
        '--action_frame_size', default=224, type=int, help='Action model input frame size')
    parser.add_argument(
        '--sceobj_frame_size', default=224, type=int, help='Scene and Object model input frame size')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=5,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--norm_value',
        default=255,
        type=int,
        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')

    # train validation
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--scheduler',
        default='warmup-cosine',
        type=str,
        help='multistep | plateau | step | warmup-cosine | warmup-multistep')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument(
        '--lr_decay', default=0.1, type=float, help='Weight Decay')
    parser.add_argument(
        '--loss_weight', default=0.5, type=float, help='Loss Weight')
    parser.add_argument(
        # '--milestones', '--arg', nargs='+', type=int, help='Milestones'
        '--milestones', default = [60], type = float, nargs = "+",
        metavar = 'LRSteps', help = 'epochs to decay learning rate by 10')
    parser.add_argument(
        '--clip', default=5.0, type=float, help='gradient clip max norm')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=True)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='sgd | adam | rmsprop | adadelta')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--plateau_thres', default=0.01, type=float, help='threshold of LR scheduler, See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--lr_decay_step', default=15, type=int, help='step_size of StepLR scheduler')
    parser.add_argument(
        '--warmup_multiplier', default=100, type=int, help='multiplier of warmup scheduler')
    parser.add_argument(
        '--warmup_epoch', default=5, type=int, help='warmup epochs')
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument(
        '--val_per_epoches',
        default=4,
        type=int,
        help='Perform validation at every this epochs.')
    parser.add_argument(
        '--checkpoint',
        default=2,
        type=int,
        help='Trained model is saved at every this epochs.')

    # other
    parser.add_argument(
        '--n_threads',
        default=8,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
