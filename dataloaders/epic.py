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
	labels = pd.read_pickle(data_file_path)
	data = []
	for i, row in labels.iterrows():
		metadata = row.to_dict()
		metadata['uid'] = i
		data.append(metadata)
	return data


def get_video_names(metadata, video_path_full):
	video_names = []
	for data in metadata:
		video_fn = '{}/{}/{}_{}_{}'.format(data['participant_id'], data['video_id'], data['video_id'],
											  data['uid'], '-'.join(data['narration']))
		video_names.append(os.path.join(video_path_full, video_fn))

	return video_names


def make_dataset(root_path, video_path, annotation_path, class_type):
	annotations = load_annotation_data(os.path.join(root_path, annotation_path))
	video_path_full = os.path.join(root_path, video_path)
	video_names = get_video_names(annotations, video_path_full)
	class_num = 125 if class_type == 'verb' else 331

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
		# ipdb.set_trace()
		temp_label = np.zeros(class_num)
		if class_type == 'verb':
			temp_label[int(annotations[i]['verb_class'])] = 1
		elif class_type == 'noun':
			temp_label[int(annotations[i]['noun_class'])] = 1

		sample['label'] = temp_label

		dataset.append(sample)
		# test_file.write(sample['video_id'] + ' ' + class_indexs + '\n')

	# test_file.close()

	return dataset


class EPIC(data.Dataset):

	def __init__(self,
				 root_path,
				 video_path,
				 annotation_path,
				 class_type='verb',
				 spatial_transform=None,
				 temporal_transform=None,
				 get_loader=get_default_video_loader):

		self.data = make_dataset(root_path, video_path, annotation_path, class_type)

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
			temp_sceobj_clip = []
			if self.spatial_transform is not None:
				self.spatial_transform.randomize_parameters()
				temp_sceobj_clip = [self.spatial_transform(img) for img in clip]

			temp_sceobj_clip = torch.stack(temp_sceobj_clip, 0)
			sceobj_clip.append(temp_sceobj_clip)

		sceobj_clip = torch.stack(sceobj_clip, 0)

		return sceobj_clip, target
