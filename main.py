## whole process
import  dbscan_cluster
import base64
import argparse
from boto3.session import Session
import json
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import cv2
import numpy as np
from io import BytesIO
import ast
import re

def invoke_glm4v(runtime, image_path, endpoint_name, collage_image=True):
    if collage_image:
        prompt='''图中是一个表格，第一列是序号，第二列是图片，按照从上往下的顺序输出以下内容：[[序号,图片中的数字],...],如果图片中没有数字则写为[[序号,"None"]...]'''
    else:
        prompt='''如果图片中有数字，输出这张图片中的数字是什么，只输出数字，不要输出其他内容；如果图片中没有数字，则输出为"None" '''
        
    image = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    im_bytes = buffer.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "role": "user",
                "image": im_b64,
                "query": prompt,
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

    return res.get('masks',None)

def sharpen_image(img_path, save_path):
    # 读取图像
    img = cv2.imread(img_path)

    # 创建卷积核
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    # 锐化处理
    sharpened = cv2.filter2D(img, -1, kernel)

    # 保存锐化后的图像
    cv2.imwrite(save_path, sharpened)

    return


def create_image_table(images, image_height, image_width, font_size=36):
    # 设置字体和字号
    font = ImageFont.load_default()
    # 计算单元格和表格尺寸
    cell_width = image_width + 20
    cell_height = image_height + 20
    num_rows = len(images)  # 向上取整
    table_width = cell_width * 2 
    table_height = cell_height * num_rows 
    # 创建表格图像和绘图对象
    table_image = Image.new("RGB", (table_width, table_height), color="white")
    draw = ImageDraw.Draw(table_image)

    # 绘制外边框
    draw.rectangle([(0, 0), (table_width - 1, table_height - 1)], outline="black", width=2)

    # 绘制单元格和插入图像
    for row in range(num_rows):
        for col in range(2):
            # 计算单元格位置
            x1 = col * (cell_width)
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            # 绘制单元格边框
            draw.rectangle([(x1, y1), (x2, y2)], outline="black")

            # 插入图像和序号
            if col==0:
                idx = row
                draw.text((x1 + 10, y1 + 10), str(idx), font=font, fill="black")
            else:
                table_image.paste(images[idx], (x1 + 10, y1 + 10))
    return table_image

## n张子图, 合并成20为整数的图, 并返回一个json
# [子图1: {[bbox1,bbox2,..,bbox20][path]}]
def create_collage(output_folder, pics, n=20):
    res = []
    input_images = [i['save_path'] for i in pics]
    num = len(input_images) // 20 + 1

    ##get image shape to the same
    image_height, image_width,_ = cv2.imread(input_images[0]).shape
    image_height+=20  # 放大尺寸，防止下面resize后，某些子图分辨率过低
    image_width+=20
    for i in tqdm(range(num)):
        # Load and resize individual images
        images = []
        #image_width = 26
        #image_height = 17
        single_image_paths=[]
        for image_path in input_images[i * 20:(i + 1) * 20]:
            if image_path.endswith(('.jpg', '.png', '.jpeg')):
                image = Image.open(image_path)
                image = image.resize((image_width, image_height))
                images.append(image)
                single_image_paths.append(image_path)
                
        if len(images)==0:
            continue
        collage_image=create_image_table(images,image_height, image_width)

#         # Create a new blank image for the collage
#         collage_width = image_width
#         collage_height = image_height * n
#         collage_image = Image.new('RGB', (collage_width, collage_height))

#         # Position and paste individual images onto the collage
#         # print ("len(image)", len(images))
#         for j in range(len(images)):
#             x = 0
#             y = j * image_height
#             collage_image.paste(images[j], (x, y))

        # Save the collage image
        if not os.path.exists(output_folder):
            # Create the directory (including parents if necessary)
            os.makedirs(output_folder, exist_ok=True)
            # print("Directory", output_folder, "created successfully!")
        # else:
            # print("Directory", output_folder, "already exists.")

        sub_name = f'{i}.jpg'
        collage_image.save(os.path.join(output_folder, sub_name))
        # print("Collage created successfully!")
        ## sharpen pics
        # sharpen_image(os.path.join(output_folder, sub_name), os.path.join(output_folder, sub_name))
        #save bbox
        merged_bboxs = [i['bbox'] for i in pics[i * 20:(i + 1) * 20]]
        res.append({'merged_image_name': os.path.join(output_folder, sub_name), 'merged_bbox': merged_bboxs, 'image_paths':single_image_paths})

    return res

def remove_non_numeric(string):
    # 使用正则表达式将字符串分割成多个部分
    parts = re.split(r'(None)', string)
    
    # 遍历每个部分
    cleaned_parts = []
    for part in parts:
        # 如果当前部分不是 "None",则使用正则表达式替换非数字字符
        if part != "None":
            cleaned_part = re.sub(r'[^0-9\.-]', '', part)
            cleaned_parts.append(cleaned_part)
        # 如果当前部分是 "None",则直接添加到结果列表中
        else:
            cleaned_parts.append(part)
    
    # 将清理后的部分连接成一个字符串
    cleaned_string = ''.join(cleaned_parts)
    
    return cleaned_string

def main(sam_endpoint_name, glm4v_endpoint_name, image_path, output_folder):
    session = Session()
    runtime = session.client("runtime.sagemaker")

    #process image
    with open(image_path, "rb") as f:
        body_base64 = base64.b64encode(f.read()).decode()
    sam_masks = invoke_sam(runtime, sam_endpoint_name, body_base64)
    if sam_masks==None:
        print('SAM gets no masks!')
        return None
    ## extract sub pics
    extract_number_pics = dbscan_cluster.ExtractNumberPics(image_path, sam_masks, dbscan_eps=20)
    ## output pics json
    pics = extract_number_pics.extract_bbox()
    results=[]
    if len(pics)/len(sam_masks)<0.1: # case 1: 图片分割和聚类效果较差,每个子图单独处理
        print('This image is predicted using a single sub-image...')
        extract_number_pics = dbscan_cluster.ExtractNumberPics(image_path, sam_masks, dbscan_eps=100)
        pics = extract_number_pics.extract_bbox()
        print('sub-image numbers:',len(pics))
        for i in range(len(pics)):
            ocr_res = invoke_glm4v(runtime, pics[i]['save_path'], glm4v_endpoint_name,collage_image=False)
            ocr_res = remove_non_numeric(ocr_res) #.replace('<|endoftext|>','').replace(' ','').replace('图片中的数字是','')
            if ocr_res=="None":
                continue
            res = {'merged_image_name': pics[i]['save_path'], 'merged_bbox': [pics[i]['bbox']],'ocr_res':[ocr_res]}
            results.append(res)
            # Image.open(pics[i]['save_path']).show()
            # print(ocr_res)
    else:
        # reget merged data
        output_dir = os.path.join('number_pics', 'merged_' + image_path.split('/')[-1].split('.')[0])
        res2 = create_collage(output_dir, pics)
        #invoke glm4v_quant
        for i in range(len(res2)):
            ocr_res = invoke_glm4v(runtime, res2[i]['merged_image_name'], glm4v_endpoint_name)
            flattened = ocr_res[:ocr_res.rfind(']')+1].replace("[", "").replace("]", "").replace(",", " ").split()
            ocr_res = flattened[1::2]
            res2[i]['ocr_res'] = ocr_res
            # Image.open(res2[i]['merged_image_name']).show()
            # print(ocr_res)
            if len(ocr_res)!= len(res2[i]['merged_bbox']):  # case 2: ocr结果数量和拼接的图片数量不符，拼接的子图单独处理
                print('This collage-image is predicted using a single sub-image, sub-image numbers: ',len(res2[i]['image_paths']))
                for j in range(len(res2[i]['image_paths'])):
                    ocr_res = invoke_glm4v(runtime, res2[i]['image_paths'][j], glm4v_endpoint_name,collage_image=False)
                    ocr_res = remove_non_numeric(ocr_res)
                    if ocr_res=="None":
                        continue
                    res = {'merged_image_name': res2[i]['image_paths'][j], 'merged_bbox': [res2[i]['merged_bbox'][j]],'ocr_res':[ocr_res]}
                    results.append(res)
                    # Image.open(res2[i]['image_paths'][j]).show()
                    # print(ocr_res)
            else:
                del res2[i]['image_paths']
                results.append(res2[i])         # case 3: perfect

    ## jsonoutput
    json_name = 'res.json'
    with open(os.path.join(output_folder, json_name), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(results, f)  # Optional parameter for indentation
    print('Data written to json',json_name)
    
    json_name_final='res_final.json'
    results_final={'merged_image_name':image_path}
    bbox_filter=[]
    ocr_filter=[]
    for merged in results:
        bbox=merged['merged_bbox']
        ocr=merged['ocr_res']
        for b,o in zip(bbox,ocr):
            o=remove_non_numeric(o)
            if len(o)>0 and "None" not in o:
                bbox_filter.append(b)
                ocr_filter.append(o)
    results_final['merged_bbox']=bbox_filter
    results_final['ocr_res']=ocr_filter
    with open(os.path.join(output_folder, json_name_final), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(results_final, f)  # Optional parameter for indentation
    print('Data written to json',json_name_final)
    
    return res2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--glm4v_endpoint_name', type=str)
    parser.add_argument('--output_folder', default = './', type=str)
    parser.add_argument('--sam_endpoint_name',type=str)
    parser.add_argument('--image_path', type=str)
    args = parser.parse_args()
    main(args.sam_endpoint_name, args.glm4v_endpoint_name, args.image_path, args.output_folder)