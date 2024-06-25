## whole process
import  dbscan_cluster
import base64
import argparse
from boto3.session import Session
import json
from PIL import Image
import os
from tqdm import tqdm
import cv2

def invoke_glm4v(runtime, im_b64, endpoint_name):
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "role": "user",
                "image": im_b64,
                "query": "从上至下图片中的数字分别是多少?",
                "gen_kwargs": {
                    "max_length": 2500,
                    "do_sample": True,
                    "top_k": 1
                },
            }
        ),
        ContentType="application/json",
    )["Body"].read().decode("utf8")
    return response

def invoke_sam(runtime, sam_endpoint_name, body_base64):
    input_data = {
        "body": body_base64,
        "task": "AutomaticMaskGenerator",
        "points_per_side": 32,
        "points_per_batch": 128,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.9,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.95,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 0.3413333333333333,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 0,
    }

    response = runtime.invoke_endpoint(
        EndpointName=sam_endpoint_name,
        ContentType="image/jpg",
        Body=json.dumps(input_data),
    )

    masks = response['Body'].read()
    res = eval(masks)['result']

    return res['masks']

def sharpen_image(img_path, save_path):
    # 读取图像
    img = cv2.imread(img_path)

    # 创建卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # 锐化处理
    sharpened = cv2.filter2D(img, -1, kernel)

    # 保存锐化后的图像
    cv2.imwrite(save_path, sharpened)

    return

## n张子图, 合并成20为整数的图, 并返回一个json
# [子图1: {[bbox1,bbox2,..,bbox20][path]}]
def create_collage(output_folder, pics, n=20):
    res = []
    input_images = [i['save_path'] for i in pics]
    num = len(input_images) // 20 + 1

    ##get image shape to the same
    image_height, image_width = cv2.imread(input_images[0]).shape

    for i in tqdm(range(num)):
        # Load and resize individual images
        images = []
        #image_width = 26
        #image_height = 17
        for image_path in input_images[i * 20:(i + 1) * 20]:
            if image_path.endswith(('.jpg', '.png', '.jpeg')):
                image = Image.open(image_path)
                image = image.resize((image_width, image_height))
                images.append(image)

        # Create a new blank image for the collage
        collage_width = image_width
        collage_height = image_height * n
        collage_image = Image.new('RGB', (collage_width, collage_height))

        # Position and paste individual images onto the collage
        # print ("len(image)", len(images))
        for j in range(len(images)):
            x = 0
            y = j * image_height
            collage_image.paste(images[j], (x, y))

        # Save the collage image
        if not os.path.exists(output_folder):
            # Create the directory (including parents if necessary)
            os.makedirs(output_folder, exist_ok=True)
            print("Directory", output_folder, "created successfully!")
        else:
            print("Directory", output_folder, "already exists.")

        sub_name = f'{i}.jpg'
        collage_image.save(os.path.join(output_folder, sub_name))
        print("Collage created successfully!")
        ## sharpen pics
        sharpen_image(os.path.join(output_folder, sub_name), os.path.join(output_folder, sub_name))
        #save bbox
        merged_bboxs = [i['bbox'] for i in pics[i * 20:(i + 1) * 20]]
        res.append({'merged_image_name': os.path.join(output_folder, sub_name), 'merged_bbox': merged_bboxs})

    return res

def main(sam_endpoint_name, glm4v_endpoint_name, image_path, output_folder):
    session = Session()
    runtime = session.client("runtime.sagemaker")

    #process image
    with open(image_path, "rb") as f:
        body_base64 = base64.b64encode(f.read()).decode()
    sam_masks = invoke_sam(runtime, sam_endpoint_name, body_base64)

    ## extract sub pics
    extract_number_pics = dbscan_cluster.ExtractNumberPics(image_path, sam_masks, dbscan_eps=20)
    ## output pics json
    pics = extract_number_pics.extract_bbox()
    # reget merged data
    output_dir = os.path.join('number_pics', 'merged_' + image_path.split('/')[-1].split('.')[0])
    res2 = create_collage(output_dir, pics)

    #invoke glm4v_quant
    for i in range(len(res2)):
        image = Image.open(res2[i]['merged_image_name']).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        im_bytes = buffer.getvalue()
        im_b64 = base64.b64encode(im_bytes).decode('utf-8')
        ocr_res = invoke_glm4v(runtime, im_b64, glm4v_endpoint_name)
        ##save
        res2[i]['ocr_res'] = ocr_res

    ## jsonoutput
    json_name = 'res.json'
    with open(os.path.join(output_folder, json_name), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(res2, f)  # Optional parameter for indentation
    print('Data written to json')
    return res2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--glm4v_endpoint_name', type=str)
    parser.add_argument('--output_folder', default = './', type=str)
    parser.add_argument('--sam_endpoint_name',type=str)
    parser.add_argument('--image_path', type=str)
    args = parser.parse_args()
    main(args.sam_endpoint_name, args.glm4v_endpoint_name, args.image_path, args.output_folder)