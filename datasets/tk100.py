'''
todo
1.将目标检测中的预测框坐标根据图片的尺寸进行缩放
'''
import torch
from torch.utils.data import Dataset
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

class TK100Dataset(Dataset):
    def __init__(self, data_info_file_path, category_path, transform=None, dataset_path=r"F:\Projects\datasets\oc\TK100\data"):
        super().__init__()

        self.data_info_file_path = data_info_file_path
        self.category_path = category_path
        self.transform = transform
        self.image_names = []
        self.box_poses = {}
        self.box_labels = {}
        self.num_images = []
        self.dataset_main_path = pathlib.Path(dataset_path)
        self.__load_oc_data()
        self.__load_oc_category()

    
    def __load_oc_data(self):
        with open(self.data_info_file_path) as f:
            lines = f.readlines()

        for line in lines:
            box_infos = line.strip().split(' ')
            self.image_names.append(box_infos[0])
            num_boxes = (len(box_infos) - 1) // 5
            box_pos = []
            box_label = []
            for i in range(num_boxes):
                lx = float(box_infos[1 + i * 5])
                ly = float(box_infos[2 + i * 5])
                rx = float(box_infos[3 + i * 5])
                ry = float(box_infos[4 + i * 5])
                label = box_infos[5 + i * 5]
                box_pos.append([lx, ly, rx, ry])
                box_label.append(label)
            self.box_poses[box_infos[0]] = box_pos
            self.box_labels[box_infos[0]] = box_label
        self.num_boxes = len(self.image_names)


    def __load_oc_category(self):
        with open(self.category_path) as f:
            categorys = f.readlines()
        self.category_2_id = {category.strip():idx for idx, category in enumerate(categorys)}
        self.id_2_category = {idx:category.strip() for idx, category in enumerate(categorys)}
        

    def __getitem__(self, index):
        '''
        对接yolov2，所以需要返回处理后的图片、预测框、预测框的类别、不关心的区域、原始图片
                # 不关心区域看代码应该是空的
        '''
        image_name = self.image_names[index]
        image_path = self.dataset_main_path / image_name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 复制一份原始图片
        image_origin = image.copy()
        image, target_boxes, target_classes, no_care_poses = self.__preprocess_train(image, self.box_poses[image_name], self.box_labels[image_name])

        
        return image, target_boxes, target_classes, no_care_poses, image_origin
        
    
    def __len__(self):
        return self.num_boxes
    

    def __preprocess_train(self, im, boxes, gt_classes):
        '''
        训练数据预处理

        param data: 待训练的图片相关数据
        param size_index: todo 未知
        '''

        # boxes： 目标预测框
        # gt_classes: 目标预测框的类别

        # 对图片进行缩放、平移、翻转等操作
        im, trans_param = self.__imcv2_affine_trans(im)
        scale, offs, flip = trans_param
        # 因为对图片进行了操作，那么预测框也要进行相应的操作
        boxes = self.__offset_boxes(boxes, im.shape, scale, offs, flip)

        # 调整图片颜色对比度等信息
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.__imcv2_recolor(im)
        # im /= 255.

        # im = imcv2_recolor(im)
        # h, w = inp_size
        # im = cv2.resize(im, (w, h))
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im /= 255
        boxes = np.asarray(boxes, dtype=np.int)
        # 进行图片增强后的图片，预测框，预测框的类别，不关心的区域
        return im, boxes, gt_classes, []
    

    def __imcv2_recolor(self, im, a=.1):
        '''
        对输入图像进行颜色扭曲和对比度调整，以达到图像增强的效果
        '''
        # t = [np.random.uniform()]
        # t += [np.random.uniform()]
        # t += [np.random.uniform()]
        # t = np.array(t) * 2. - 1.
        t = np.random.uniform(-1, 1, 3)

        # random amplify each channel
        # 进行颜色增强
        im = im.astype(np.float)
        im *= (1 + t * a)
        # 对比度调整
        mx = 255. * (1 + a)
        up = np.random.uniform(-1, 1)
        im = np.power(im / mx, 1. + up * .5)
        # return np.array(im * 255., np.uint8)
        return im


    def __offset_boxes(self, boxes, im_shape, scale, offs, flip):
        '''
        根据对图片的缩放、平移、翻转等操作，将目标预测框进行偏移

        param boxes: 预测框列表
        param im_shape: 输入的图片大小（已经转换后的）
        param scale: 缩放比例
        param offs: 偏移量
        param flip: 是否翻转
        '''

        if len(boxes) == 0:
            return boxes
        boxes = np.asarray(boxes, dtype=np.float)
        boxes *= scale
        # 对预测框偏移
        # todo 这里是怎样进行偏移
        boxes[:, 0::2] -= offs[0]
        boxes[:, 1::2] -= offs[1]
        # 如果预测框偏移到边界外，则将其置为边界
        boxes = self.__clip_boxes(boxes, im_shape)

        # todo 这里是图和进行图片翻转的操作
        if flip:
            boxes_x = np.copy(boxes[:, 0])
            boxes[:, 0] = im_shape[1] - boxes[:, 2]
            boxes[:, 2] = im_shape[1] - boxes_x

        return boxes


    def __clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes


    def __imcv2_affine_trans(self, im):
        '''
        将图片进行所缩放、平移、翻转等操作，也就是图像增强

        param im: 原始图片数据
        '''
        # Scale and translate
        h, w, c = im.shape
        scale = np.random.uniform() / 10. + 1. # 计算缩放比例
        max_offx = (scale - 1.) * w # 计算x轴最大偏移量，(缩放比例-1) = 缩放后尺寸/原始尺寸
        max_offy = (scale - 1.) * h # 缩放比例-1代表缩放后的尺寸和原始尺寸的比例差值
        offx = int(np.random.uniform() * max_offx) # 随机生成x轴偏移量，不超过最大的偏移值
        offy = int(np.random.uniform() * max_offy) # 随机生成y轴偏移量，不超过最大的偏移值

        im = cv2.resize(im, (0, 0), fx=scale, fy=scale) # 缩放图片
        im = im[offy: (offy + h), offx: (offx + w)] # 偏移图片
        flip = np.random.uniform() > 0.5 # 随机生成是否翻转图片的标志
        if flip:
            im = cv2.flip(im, 1)

        return im, [scale, [offx, offy], flip]
    

if __name__ == "__main__":
    data_info_file_path = r'M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100.txt'
    category_path = r'M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100-catetory.txt'
    tk100_dataset = TK100Dataset(data_info_file_path, category_path)
    # 使用matplotlib显示图片，并将预测框绘制再图片上
     # 随机选择一个样本
    idx = np.random.randint(0, len(tk100_dataset))
    image, box_poses, category_ids, _, image_origin = tk100_dataset[idx]

    # 要使得image能够显示，需要去掉__imcv2_recolor避免图片被转换为float
    # 将图片从RGB转换为BGR (因为OpenCV默认是BGR) 
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 创建figure
    fig, ax = plt.subplots(1)

    # 显示图片
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 绘制预测框
    for box, label in zip(box_poses, category_ids):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        print(f"label: {tk100_dataset.id_2_category[int(label)]}")

    # 获取类别名称
    # 关闭坐标轴
    plt.axis('off')

    # 显示图片
    plt.show()

    # 打印一些信息
    print(f"Image shape: {image_origin.shape}")
    print(f"Number of boxes: {len(box_poses)}")
