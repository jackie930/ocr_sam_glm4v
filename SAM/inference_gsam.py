import os
import glob, subprocess
import boto3
import base64
import uuid
import io
import json
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision

import argparse
import copy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2
import supervision as sv

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import pickle
import gzip

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

NMS_THRESHOLD = 0.8

BOX_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.8

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def model_fn():
    logger.info('heather /opt/ml/code/')

    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor


        
def predict_fn(input_json, model):
    print("heather request body: ", input_json)

    grounding_dino_model,sam_predictor=model_fn()


    if grounding_dino_model is None or sam_predictor is None:
        print("Models could not be initialized. Exiting...")
        return
    request_id = input_json['request_id'] if 'request_id' in input_json.keys() else "notdefined"
    input_image_in_b64 = input_json['input_image']
    input_image_in_b64 = input_image_in_b64.encode()
    input_image_in = base64.b64decode(input_image_in_b64)
    filename_input = str(uuid.uuid4())
    fh = open(f"assets/{filename_input}.png", "wb+")
    fh.write(input_image_in)
    fh.close()

    box_threshold_in = input_json['box_threshold']
    text_threshold_in = input_json['text_threshold']
    text_prompt_in = input_json['text_prompt']
    CLASSES=text_prompt_in
    s3_output_dir = input_json['s3_output_dir']

    image = cv2.imread(f"assets/{filename_input}.png")
    detections = grounding_dino_model.predict_with_classes(image=image,
                                                               classes=CLASSES,
                                                               box_threshold=float(box_threshold_in),
                                                               text_threshold=float(text_threshold_in))

    
    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")
        
    tmp_mask = detections.mask
    b = pickle.dumps(tmp_mask)
    ret = gzip.compress(b)
    with open(f"./outputs/mask_{request_id}.txt","wb+") as f:
        f.write(ret)
    index = 0
    for mask in detections.mask:
        mask_image=show_mask_cv2(mask, random_color=False)
        mask_image_new = cv2.convertScaleAbs(mask_image, alpha=(255.0))
        print(mask_image_new.shape)
        cv2.imwrite(f"./outputs/grounded_sam_annotated_image_{request_id}_{index}.jpg", mask_image_new)
        index += 1

    print("successfull -- sam image")
    
    
    s3 = boto3.client('s3')
    result_ls = glob.glob('outputs/*')
    logger.info(result_ls)
    for each in result_ls:
        filename = os.path.split(each)[-1]
        s3.upload_file(each, s3_output_dir, os.path.join("gsam",filename))
    resultjson = json.dumps(
            {
                'result': result_ls
            }
        )
    print("heather print: results")
    print(result_ls)
    os.remove(f"assets/{filename_input}.png")
    print(f"delete assets/{filename_input}.png")
    return resultjson


