# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import atexit
import bisect
import glob
import multiprocessing as mp
import numpy as np
import os
import time
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.utils.logger import setup_logger

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.freeze()
	return cfg


def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
	parser.add_argument(
		"--config-file",
		default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
	parser.add_argument("--video-input", help="Path to video file.")
	parser.add_argument(
		"--input",
		nargs="+",
		help="A list of space separated input images; "
			 "or a single glob pattern such as 'directory/*.jpg'",
	)
	parser.add_argument(
		"--output",
		help="A file or directory to save output visualizations. "
			 "If not given, will show output in an OpenCV window.",
	)

	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	parser.add_argument("--parallel", action="store_true", help="Whether to run the model in different processes")
	parser.add_argument("--gpu_id", type=str, default='0', help="GPU id to use")
	return parser


class AsyncPredictor:
	"""
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

	class _StopToken:
		pass

	class _PredictWorker(mp.Process):
		def __init__(self, cfg, task_queue, result_queue):
			self.cfg = cfg
			self.task_queue = task_queue
			self.result_queue = result_queue
			super().__init__()

		def run(self):
			predictor = Predictor(self.cfg)

			while True:
				task = self.task_queue.get()
				if isinstance(task, AsyncPredictor._StopToken):
					break
				idx, data = task
				result = predictor(data)
				self.result_queue.put((idx, result))

	def __init__(self, cfg, num_gpus: int = 1):
		"""
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
		num_workers = max(num_gpus, 1)
		self.task_queue = mp.Queue(maxsize=num_workers * 3)
		self.result_queue = mp.Queue(maxsize=num_workers * 3)
		self.procs = []
		for gpuid in range(max(num_gpus, 1)):
			cfg = cfg.clone()
			cfg.defrost()
			cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
			self.procs.append(
				AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
			)

		self.put_idx = 0
		self.get_idx = 0
		self.result_rank = []
		self.result_data = []

		for p in self.procs:
			p.start()
		atexit.register(self.shutdown)

	def put(self, image):
		self.put_idx += 1
		self.task_queue.put((self.put_idx, image))

	def get(self):
		self.get_idx += 1  # the index needed for this request
		if len(self.result_rank) and self.result_rank[0] == self.get_idx:
			res = self.result_data[0]
			del self.result_data[0], self.result_rank[0]
			return res

		while True:
			# make sure the results are returned in the correct order
			idx, res = self.result_queue.get()
			if idx == self.get_idx:
				return res
			insert = bisect.bisect(self.result_rank, idx)
			self.result_rank.insert(insert, idx)
			self.result_data.insert(insert, res)

	def __len__(self):
		return self.put_idx - self.get_idx

	def __call__(self, image):
		self.put(image)
		return self.get()

	def shutdown(self):
		for _ in self.procs:
			self.task_queue.put(AsyncPredictor._StopToken())

	@property
	def default_buffer_size(self):
		return len(self.procs) * 5


class Predictor(DefaultPredictor):
	def __call__(self, original_image):
		"""
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
		with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
			# Apply pre-processing to image.
			if self.input_format == "RGB":
				# whether the model expects BGR inputs or RGB
				original_image = original_image[:, :, ::-1]
			height, width = original_image.shape[:2]
			image = self.transform_gen.get_transform(original_image).apply_image(original_image)
			image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

			inputs = {"image": image, "height": height, "width": width}
			images = self.model.preprocess_image([inputs])
			features = self.model.backbone(images.tensor)
			proposals, _ = self.model.proposal_generator(images, features)
			instances = self.model.roi_heads._forward_box(features, proposals)

			# if no object was detected, then use whole image as bbox
			if instances[0].scores.shape[0] == 0:
				w, h = instances[0].image_size
				tbox = torch.from_numpy(np.array([0, 0, w, h]))
				tbox = tbox.view(1, -1).to(instances[0].pred_boxes.tensor.device,
										   instances[0].pred_boxes.tensor.dtype)
				instances[0].pred_boxes.tensor = torch.cat((instances[0].pred_boxes.tensor, tbox), dim=0)

			features = [features[f] for f in self.model.roi_heads.in_features]
			box_features = self.get_box_features(features, [x.pred_boxes for x in instances])
			mask_features = self.get_mask_features(features, [x.pred_boxes for x in instances])

			for box_feature, mask_feature, instance in zip([box_features], [mask_features], instances):
				instance.box_features = box_feature
				instance.mask_features = mask_feature

			return instances

	def get_box_features(self, features, boxes):
		box_features = self.model.roi_heads.box_pooler(features, boxes)
		box_features = self.model.roi_heads.box_head(box_features)

		return box_features

	def get_mask_features(self, features, boxes):
		mask_features = self.model.roi_heads.mask_pooler(features, boxes)
		mask_features = self.model.roi_heads.box_head(mask_features)

		return mask_features


if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	cfg = setup_cfg(args)

	if args.parallel:
		num_gpu = torch.cuda.device_count()
		predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
	else:
		predictor = Predictor(cfg)

	if args.input:
		for img_dir in os.listdir(args.input[0]):
			imgs = glob.glob(os.path.expanduser(os.path.join(args.input[0], img_dir, '*/*.jpg')))
			for path in tqdm.tqdm(imgs, disable=not args.output):
				dirname, basename = os.path.split(path)
				output_dir = dirname.replace('videos_256x256_30', 'features')
				os.makedirs(output_dir) if not os.path.exists(output_dir) else None
				out_filename = os.path.join(output_dir, os.path.splitext(basename)[0]) + '.npz'
				if os.path.exists(out_filename):
					if os.stat(out_filename).st_size > 536:
						continue

				# use PIL, to be consistent with evaluation
				img = read_image(path, format="BGR")
				start_time = time.time()
				predictions = predictor(img)[0]
				logger.info(
					"{}: {} in {:.2f}s".format(
						path,
						"detected {} instances".format(predictions.scores.shape[0]),
						time.time() - start_time,
					)
				)

				with open(out_filename, 'wb') as f:
					np.savez(f, box_features=predictions.box_features.cpu().numpy(),
							 mask_features=predictions.mask_features.cpu().numpy())
