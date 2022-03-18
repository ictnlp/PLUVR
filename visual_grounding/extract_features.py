import os
import cv2
import sys
import glob
import json
import torch
import argparse
import numpy as np
from os.path import abspath, dirname, join

root = dirname(dirname(abspath(__file__)))
detectron_path = join(root, 'detectron2')
model_weights_path = join(detectron_path, 'model_weights')
dataset_path = join(root, 'dataset/multi30k')
images_path = join(dataset_path, 'data/raw_images/flickr30k-images')
files = join(root, 'visual_grounding/files')

sys.path.insert(0, join(root, 'detectron2'))

import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, Boxes


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
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
    return parser


def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    model = build_model(cfg)
    model.eval()
    weights = os.path.join(model_weights_path, 'faster_rcnn_R_101_C4_3x.pkl')
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(weights)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    input_format = cfg.INPUT.FORMAT

    with open(join(files, 'bbox.json'), 'r') as f:
        bboxes = json.load(f)

    results = []

    def hook_fn_forward(module, input, output):
        feature = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        results.append(feature[0, :, 0, 0].cpu().numpy().tolist())

    model.roi_heads.res5[2].conv3.register_forward_hook(hook_fn_forward)

    with torch.no_grad():
        chunk_size = 10000
        for chunk_id in range(len(bboxes) // chunk_size + 1):
            begin = chunk_id * chunk_size
            end = min((chunk_id + 1) * chunk_size, len(bboxes))
            results = []
            for idx in range(begin, end):
                bbox = bboxes[idx]
                print('{0}/{1}'.format(idx, len(bboxes)))
                image_path = join(images_path, bbox['image'])
                original_image = read_image(image_path, format='BGR')
                pred_boxes = Boxes(torch.tensor(bbox['bbox']).unsqueeze(0))
                pred_classes = torch.zeros(pred_boxes.tensor.size(0), dtype=torch.long)

                if input_format == 'RGB':
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))

                inputs = {'image': image, 'height': height, 'width': width}
                detected_instances = Instances((height, width), pred_boxes=pred_boxes, pred_classes=pred_classes)

                predictions = model.inference([inputs], detected_instances=[detected_instances])[0]
    
            results = np.array(results)
            filename = 'bbox_features_faster_rcnn_R_101_' + '%02d' % chunk_id + '.npy'
            np.save(join(files, filename), results)


if __name__ == '__main__':
    main()