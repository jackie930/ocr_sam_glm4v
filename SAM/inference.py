import os
import sys
import json
import io
import base64
import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import boto3
import uuid
import requests
import traceback
from utils import check_weights, load_model_hf_grounddino
from groundingdino.util.inference import Model
import lzstring
from pycocotools import mask
import torchvision
import pickle
import gzip

BASE_DIR = os.path.expanduser("~/.sam/")
s3_client = boto3.client('s3')
lzstring_processor = lzstring.LZString()
print("<<< start jackie!!!")


def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key


def base64_to_bytes(base64_message):
    base64_img_bytes = base64_message.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    return decoded_image_data


def model_fn(model_dir):
    """
    Load the model for inference
    """

    model_type = os.environ['model_type']
    print('model_type: ', model_type)


    if model_type != 'sam_vit_h' and model_type != 'sam_vit_l':
        model_type = 'sam_vit_b'

    print('model_type final: ', model_type)
    sam_checkpoint = check_weights(model_type, model_dir, BASE_DIR)
    sam_checkpoint_type = model_type[4:]

    if torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA.")
    else:
        device = 'cpu'
        print("CUDA is not available. Using CPU.")

    sam = sam_model_registry[sam_checkpoint_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("<<<< load sam success!!!")
    sam_predictor = SamPredictor(sam)

    # load grounddino
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH = load_model_hf_grounddino(ckpt_repo_id, ckpt_filenmae,
                                                                                          ckpt_config_filename, device)
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    print("<<<< load grounddino success!!!")

    return sam, grounding_dino_model, sam_predictor


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == 'image/jpg' or request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        data = request_body
        data = json.loads(data)
        bytes_base64 = data['body']
        bytes = base64_to_bytes(bytes_base64)
        image = Image.open(io.BytesIO(bytes))
        data['image'] = image
        return data
    elif request_content_type == 'application/json':
        data = request_body
        data = json.loads(data)
        bucket = data['bucket']
        image_uri = data['image_uri']
        s3_object = s3_client.get_object(Bucket=bucket, Key=image_uri)
        bytes = s3_object["Body"].read()
        image = Image.open(io.BytesIO(bytes))
        data['image'] = image
        return data
    else:
        return request_body


def segment(sam_predictor, image, xyxy):
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


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    print('Input_data Key:')
    print(input_data.keys())

    image = input_data['image']
    ## 判读图片类型
    if image.mode == 'RGB' or image.mode == 'RGBA':
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        pil_image = image.convert('L')
        # Convert to NumPy array
        np_image = np.array(pil_image)
        # Now you can use np_image with OpenCV functions
        open_cv_image = np_image  # No need for color conversion
        image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

    print(image.shape)
    task = input_data['task']
    print('Task: ', task)
    prediction = {}
    prediction['task'] = task

    try:
        if task == 'AutomaticMaskGenerator':
            # sam parameter
            prediction['points_per_side'] = int(input_data['points_per_side'])
            prediction['points_per_batch'] = int(input_data['points_per_batch'])
            prediction['pred_iou_thresh'] = float(input_data['pred_iou_thresh'])
            prediction['stability_score_thresh'] = float(input_data['stability_score_thresh'])
            prediction['stability_score_offset'] = float(input_data['stability_score_offset'])
            prediction['box_nms_thresh'] = float(input_data['box_nms_thresh'])
            prediction['crop_n_layers'] = int(input_data['crop_n_layers'])
            prediction['crop_nms_thresh'] = float(input_data['crop_nms_thresh'])
            prediction['crop_overlap_ratio'] = float(input_data['crop_overlap_ratio'])
            prediction['crop_n_points_downscale_factor'] = int(input_data['crop_n_points_downscale_factor'])
            prediction['min_mask_region_area'] = int(input_data['min_mask_region_area'])

            mask_generator = SamAutomaticMaskGenerator(
                model=model[0],
                points_per_side=prediction['points_per_side'],
                points_per_batch=prediction['points_per_batch'],
                pred_iou_thresh=prediction['pred_iou_thresh'],
                stability_score_thresh=prediction['stability_score_thresh'],
                stability_score_offset=prediction['stability_score_offset'],
                box_nms_thresh=prediction['box_nms_thresh'],
                crop_n_layers=prediction['crop_n_layers'],
                crop_nms_thresh=prediction['crop_nms_thresh'],
                crop_overlap_ratio=prediction['crop_overlap_ratio'],
                crop_n_points_downscale_factor=prediction['crop_n_points_downscale_factor'],
                min_mask_region_area=prediction['min_mask_region_area'],
            )
            masks = mask_generator.generate(image)
            print ("<<<< number of masks: ", len(masks))

            # output no segmentation result
            '''
            print("start convert numpy to list!")
            for i in masks:
                mask_np = np.asfortranarray(i['segmentation'])
                rle = mask.encode(mask_np)
                ret = lzstring_processor.compressToEncodedURIComponent(rle["counts"].decode())
                i['segmentation'] = ret
            '''
            for i in masks:
                del i["segmentation"]

            prediction['masks'] = masks
            prediction['task'] = task

        elif task == 'Predictor':
            print('In Task')
            predictor = SamPredictor(model[0])

            print('before set image')
            predictor.set_image(image)
            print('after set image')
            point_coords = None
            point_labels = None
            box = None
            mask_input = None
            multimask_output = False

            print(input_data.keys())

            if 'point_coords' in input_data.keys():
                if input_data['point_coords'] != '':
                    point_coords = np.array(input_data['point_coords'])
                    print('point_coords:', point_coords)

            if 'point_labels' in input_data.keys():
                if input_data['point_labels'] != '':
                    point_labels = np.array(input_data['point_labels'])
                    print('point_labels: ', point_labels)

            if 'box' in input_data.keys():
                if input_data['box'] != '':
                    box = np.array(input_data['box'])
                    print('box: ', box)

            if 'mask_input' in input_data.keys():
                if input_data['mask_input'] != '':
                    mask_input = np.array(input_data['mask_input'])

            if 'multimask_output' in input_data.keys():
                if input_data['mask_input'] == 1:
                    multimask_output = True

            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )

            prediction['masks'] = masks.tolist()
            prediction['scores'] = scores.tolist()
            prediction['logits'] = logits.tolist()
        elif task == 'Remove':
            print('In Task')
            predictor = SamPredictor(model[0])

            print('before set image')
            predictor.set_image(image)
            print('after set image')
            point_coords = None
            point_labels = None
            box = None
            mask_input = None
            multimask_output = True

            print(input_data.keys())

            if 'point_coords' in input_data.keys():
                if input_data['point_coords'] != '':
                    point_coords = np.array(input_data['point_coords'])
                    print('point_coords:', point_coords)

            if 'point_labels' in input_data.keys():
                if input_data['point_labels'] != '':
                    point_labels = np.array(input_data['point_labels'])
                    print('point_labels: ', point_labels)

            if 'box' in input_data.keys():
                if input_data['box'] != '':
                    box = np.array(input_data['box'])
                    print('box: ', box)

            if 'mask_input' in input_data.keys():
                if input_data['mask_input'] != '':
                    mask_input = np.array(input_data['mask_input'])

            if 'multimask_output' in input_data.keys():
                if input_data['mask_input'] == 0:
                    multimask_output = False

            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )
            sum_list = list(map(lambda x: sum(np.reshape(x, [-1])), masks))
            mask_index = sum_list.index(max(sum_list))

            prediction['masks'] = masks[mask_index].tolist()
            prediction['scores'] = scores.tolist()
            prediction['logits'] = logits.tolist()
        elif task == 'ImageEmb':
            print('In Task')
            predictor = SamPredictor(model[0])
            print('before set image')
            predictor.set_image(image)
            print('after set image')
            prediction['original_size'] = predictor.original_size
            prediction['input_size'] = predictor.input_size
            print("process features")
            prediction['features'] = predictor.features.cpu().numpy().tolist()
            print("process features successfully")
            prediction['is_image_set'] = predictor.is_image_set
        elif task == 'gsam':
            ##get grounddino bbox
            box_threshold_in = float(input_data['box_threshold'])
            text_threshold_in = float(input_data['text_threshold'])
            text_prompt_in = str(input_data['text_prompt'])
            classes = [text_prompt_in]
            grounding_dino_model = model[1]
            detections = grounding_dino_model.predict_with_classes(image=image,
                                                                   classes=classes,
                                                                   box_threshold=box_threshold_in,
                                                                   text_threshold=text_threshold_in)
            # NMS post process
            print(f"Before NMS: {len(detections.xyxy)} boxes")
            NMS_THRESHOLD = 0.8
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # convert detections to masks
            sam_predictor = model[2]
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy)

            # output and compress
            print("<<< detections after process success! ", detections.mask)
            # b = pickle.dumps(detections.mask)
            # prediction = gzip.compress(b)

            #compress
            mask_candidate = detections.mask[0]
            mask_np = np.asfortranarray(mask_candidate)
            rle = mask.encode(mask_np)
            ret = lzstring_processor.compressToEncodedURIComponent(rle["counts"].decode())
            print("<<< detections after compress", ret)
            prediction['masks'] = ret

        else:
            # TODO: Batch Transform
            print("Task Input Type Error: ", task)

    except Exception as e:
        print(e)
        traceback.print_exc()

    #print('prediction size: ', sys.getsizeof(prediction) / 1024)
    return prediction


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    print('Go to output fn')
    return json.dumps(
        {
            'result': prediction
        }
    )
