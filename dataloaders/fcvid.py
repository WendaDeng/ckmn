import torch
import torch.utils.data as data

from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
from utils import load_value_file, load_list_file
import jpeg4py as jpeg


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        # image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        image_path = os.path.join(video_dir_path, i)
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            #label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key)
            annotations.append(value['annotations'])

    return video_names, annotations

def make_dataset(root_path, annotation_path, subset):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    dataset = []
    #test_file = open('/DATA/disk1/qzb/datasets/FCVID/test_files_' + subset + '.txt', 'w')
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'number_Frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue
        frame_indices_file_path = os.path.join(video_path, 'frames_name')
        frame_indices = load_list_file(frame_indices_file_path)

        sample = {
            'video': video_path,
            'n_frames': n_frames,
            'frame_indices': frame_indices,
            'video_id': video_names[i]
        }
        #ipdb.set_trace()
        class_indexs = annotations[i]['label']
        if not '-' in class_indexs:
            temp_label = np.zeros(len(class_to_idx))
            temp_label[int(class_indexs)] = 1

            sample['label'] = temp_label
        else:
            temp = class_indexs.split('-')
            temp_label = np.zeros(len(class_to_idx))
            temp_label[int(temp[0])] = 1

            #for class_index in temp:
            #    temp_label[int(class_index)] = 1

            sample['label'] = temp_label
        dataset.append(sample)
        #test_file.write(sample['video_id'] + ' ' + class_indexs + '\n')

    #test_file.close()    

    return dataset


class FCVID(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):

        self.data = make_dataset(root_path, annotation_path, subset)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        target = self.data[index]['label']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        path = self.data[index]['video']
        sceobj_clip = []
        for i in range(len(frame_indices)):
            clip = self.loader(path, frame_indices[i])

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                temp_sceobj_clip = [self.spatial_transform(img) for img in clip]

            temp_sceobj_clip = torch.stack(temp_sceobj_clip, 0)
            sceobj_clip.append(temp_sceobj_clip)

        sceobj_clip = torch.stack(sceobj_clip, 0)

        return sceobj_clip, target
