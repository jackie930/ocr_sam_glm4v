from sklearn.cluster import DBSCAN
from collections import Counter
import os
import cv2
from io import BytesIO
import base64
from PIL import Image
import numpy as np




def np_to_mask_candidate(mask_np):
    """
    将 NumPy 数组转换回原始的 mask_candidate 格式

    Args:
        mask_np (numpy.ndarray): NumPy 数组形式的掩码

    Returns:
        list: mask_candidate 列表
    """
    # 确保输入是一维数组
    mask_np = np.ravel(mask_np)

    # 将数组转换为字节串
    mask_bytes = np.packbits(mask_np).tobytes()

    # 将字节串转换为列表
    mask_candidate = [None] * (len(mask_bytes) * 8)
    for i, byte in enumerate(mask_bytes):
        for j in range(8):
            mask_candidate[i * 8 + j] = bool(byte & (1 << (7 - j)))

    return mask_candidate

class ExtractNumberPics():
    def __init__(self, image_path, sam_masks,save_seg_pic=True,dbscan_eps=20):
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

    def extract_mask_image(self, bbox):
        # 获取边界框区域
        x, y, width, height = bbox
        roi = self.image[y:y+height, x:x+width]
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
            os.makedirs(os.path.join('number_pics',self.image_path.split('/')[-1].split('.')[0]),exist_ok=True)

        for i in range(len(select_masks)):
            pic_info={}
            pic_info['bbox']=[int(i) for i in select_masks[i]['bbox']]

            mask_image = self.extract_mask_image(pic_info['bbox'])
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
