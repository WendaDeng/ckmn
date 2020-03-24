import torch
import torch.utils.data as data

from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
import pandas as pd
import random
from skimage import io

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
import ipdb


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


def load_annotation_data(data_file_path, dataset_break=5):
	labels = pd.read_pickle(data_file_path)
	data = []
	j = 0
	for i, row in labels.iterrows():
		if j % dataset_break == 0:
			metadata = row.to_dict()
			metadata['uid'] = i
			data.append(metadata)
		j += 1
	return data


def get_video_names(metadata, video_path_full, mode='train'):
	video_names = []

	if mode == 'test':
		for data in metadata:
			video_fn = '{}/{}/{}_{}_{}'.format(data['participant_id'], data['video_id'], data['video_id'],
												  data['uid'], 'unnarrated')
			video_names.append(os.path.join(video_path_full, video_fn))
	else:
		for data in metadata:
			video_fn = '{}/{}/{}_{}_{}'.format(data['participant_id'], data['video_id'], data['video_id'],
												  data['uid'], data['narration'].replace(' ', '-'))
			video_names.append(os.path.join(video_path_full, video_fn))

	return video_names


def make_dataset(video_path, annotation_path, dataset_break, mode='train'):
	annotations = load_annotation_data(annotation_path, dataset_break)
	video_names = get_video_names(annotations, video_path, mode)

	print('len of video_names', len(video_names))
	dataset = []
	# test_file = open('/DATA/disk1/qzb/datasets/FCVID/test_files_' + subset + '.txt', 'w')
	for i in range(len(video_names)):
		if i % 1000 == 0:
			print('dataset loading [{}/{}]'.format(i, len(video_names)))

		if not os.path.exists(video_names[i]):
			print(video_names[i])
			continue

		frame_indices = os.listdir(video_names[i])
		n_frames = len(frame_indices)
		if n_frames <= 0:
			continue

		sample = {
			'video': video_names[i],
			'n_frames': n_frames,
			'frame_indices': frame_indices
		}

		if 'verb_class' in annotations[i]:
			sample['label'] = {'verb': annotations[i]['verb_class'], 'noun': annotations[i]['noun_class']}
		else:	# Fake label to deal with the test sets (S1/S2) that dont have any labels
			sample['label'] = -10000

		dataset.append(sample)
		# test_file.write(sample['video_id'] + ' ' + class_indexs + '\n')

	# test_file.close()
	print('len of dataset', len(dataset))
	return dataset


class EPIC(data.Dataset):

	def __init__(self,
				 video_path,
				 annotation_path,
				 mode='train',
				 object_feature_path=None,
				 dataset_break=5,
				 spatial_transform=None,
				 temporal_transform=None,
				 get_loader=get_default_video_loader,
				 obj_feature_type=None,
				 use_obj_feature=False):

		self.mode = mode
		self.data = make_dataset(video_path, annotation_path, dataset_break, self.mode)

		self.spatial_transform = spatial_transform
		self.temporal_transform = temporal_transform
		self.loader = get_loader()
		self.video_path = video_path
		self.object_feature_path = object_feature_path
		self.use_obj_feature = use_obj_feature
		self.obj_feature_type = obj_feature_type

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
			temp_sceobj_clip = []
			if self.spatial_transform is not None:
				self.spatial_transform.randomize_parameters()
				temp_sceobj_clip = [self.spatial_transform(img) for img in clip]

			temp_sceobj_clip = torch.stack(temp_sceobj_clip, 0)
			sceobj_clip.append(temp_sceobj_clip)
		sceobj_clip = torch.stack(sceobj_clip, 0)

		if self.use_obj_feature:
			obj_features = []
			for i in range(len(frame_indices)):
				path = path.replace(self.video_path, self.object_feature_path)
				obj_feature = self.load_feature(path, frame_indices[i], self.obj_feature_type)
				obj_features.append(obj_feature)

			obj_features = torch.stack(obj_features, 0)
			return sceobj_clip, target, obj_features

		return sceobj_clip, target

	def load_feature(self, feature_path, frame_indices, obj_feature_type='box'):
		"""
		:param feature_path: path that store object features
		:param frame_indices: ['frame_0000000001.jpg', 'frame_0000000002.jpg', ...]
		:param obj_feature_type: 'box' or 'mask'
		:return: tensor which has shape of [N, 1024], N is total number of objects in all frames
		 """
		features = []
		for i in frame_indices:
			npz_file = os.path.join(feature_path, i.replace('.jpg', '.npz'))
			if os.path.exists(npz_file):
				npzfile = np.load(npz_file)
				if obj_feature_type == 'box':
					feature = torch.from_numpy(npzfile['box_features'])
				elif obj_feature_type == 'mask':
					feature = torch.from_numpy(npzfile['mask_features'])
				features.append(feature)
		features = torch.cat(features, dim=0)
		feature, _ = torch.max(features, dim=0)

		return feature
