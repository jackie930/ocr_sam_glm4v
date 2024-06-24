from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
import os
import cv2
from io import BytesIO
import base64
from PIL import Image

class ExtractNumberPics():
    def __init__(self, image_path, sam_masks,save_seg_pic=False,dbscan_eps=20):
        """
        聚类后，输出含有数字的SAM结果
        1. image:读入的图片，格式为：
        image = cv2.imread('爆炸图_双桶_下排水_进水选择_MTW100-P1101Q-AE_产品爆炸图.JPG')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        2. sam_masks : SAM分割后的结果，如：
        mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=64)
        masks = mask_generator.generate(image)
        3. dbscan_eps， DBSAN聚类参数，数字越小，分出的类别数量越多
        
        example：
        import  dbscan_cluster
        extract_number_pics=dbscan_cluster.ExtractNumberPics(image,sam_masks,dbscan_eps=20)
        pics=extract_number_pics.extract_bbox()

        for i  in range(len(pics)):
            mask_image=pics[i]
            cv2.imwrite("./number_pics/pic_"+str(i)+".jpg", cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))  
        """
        self.image_path=image_path
        self.sam_masks=sam_masks
        self.dbscan_eps=dbscan_eps
        self.save_seg_pic=save_seg_pic
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def cluster_integers(self,numbers):
        """
        对给定的整数列表进行聚类,返回每个整数所属的类别
        :param numbers: 包含整数的列表
        :return: 每个整数所属的类别
        """
        # 将整数列表转换为NumPy数组
        X = np.array(numbers).reshape(-1, 1)

        # 创建DBSCAN对象
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=2)

        # 执行聚类
        labels = dbscan.fit_predict(X)

        # 返回每个整数所属的类别
        return labels
    
    def extract_mask_image(self,mask, bbox):
        """
        从原始图像中提取掩码区域和边界框内的区域,并生成新的图像。

        参数:
        image (numpy.ndarray): 原始图像(OpenCV格式)
        mask (numpy.ndarray): 二维掩码,形状与原始图像相同,元素为布尔值(True或False)
        bbox (list): 边界框坐标,格式为[x, y, width, height]

        返回:
        numpy.ndarray: 提取后的新图像(OpenCV格式)
        """
        # 创建新的图像数组,只包含掩码区域和边界框内的像素
        mask_image_array = np.zeros_like(self.image)
        mask_image_array[mask] = self.image[mask]

        # 获取边界框区域
        x, y, width, height = bbox
        roi = mask_image_array[y:y+height, x:x+width]

        return roi
    
    def extract_bbox(self):
        areas=[i['area'] for i in self.sam_masks]
        labels = self.cluster_integers(areas)
        cc=Counter(labels)
        # print(cc)
        cc.pop(-1)
        select_label = max(cc.items(), key=lambda x: x[1])[0]
        # print('select label:',select_label)
        select_masks=[i for i,j in zip(self.sam_masks,labels) if j==select_label]
        # print('select bbox:',len(select_masks))
        
        select_pics=[]
        if self.save_seg_pic:
            os.makedirs('number_pics',exist_ok=True)
            os.makedirs(self.image_path.split('/')[-1].split('.')[0],exist_ok=True)
            
        for i in range(len(select_masks)):
            pic_info={}
            pic_info['bbox']=[int(i) for i in select_masks[i]['bbox']]
            
            mask_image = self.extract_mask_image(select_masks[i]['segmentation'],pic_info['bbox'])
            buffer = BytesIO()
            pil_image = Image.fromarray(np.asarray(mask_image))
            pil_image.save(buffer, format="JPEG")
            im_bytes = buffer.getvalue()
            im_b64 = base64.b64encode(im_bytes).decode('utf-8')
            pic_info['im_b64']=im_b64
            
            if self.save_seg_pic:
                pic_info['save_path']=os.path.join("number_pics",self.image_path.split('/')[-1].split('.')[0],str(i)+".jpg")
                cv2.imwrite(pic_info['save_path'], cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
            select_pics.append(pic_info)
  
        return select_pics
