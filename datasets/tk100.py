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
        if self.transform:
            image = self.transform(image)
        
        return image, self.box_poses[image_name], self.category_2_id[self.box_labels[image_name]], [], image_origin
        
    
    def __len__(self):
        return self.num_boxes
    

if __name__ == "__main__":
    data_info_file_path = r'M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100.txt'
    category_path = r'M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100-catetory.txt'
    tk100_dataset = TK100Dataset(data_info_file_path, category_path)
    # 使用matplotlib显示图片，并将预测框绘制再图片上
     # 随机选择一个样本
    idx = np.random.randint(0, len(tk100_dataset))
    image, box_poses, category_id, _, image_origin = tk100_dataset[idx]

    # 将图片从RGB转换为BGR (因为OpenCV默认是BGR)
    image_origin = cv2.cvtColor(image_origin, cv2.COLOR_RGB2BGR)

    # 创建figure
    fig, ax = plt.subplots(1)

    # 显示图片
    ax.imshow(cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB))

    # 绘制预测框
    for box in box_poses:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    # 获取类别名称
    category_name = tk100_dataset.id_2_category[category_id]

    # 设置标题
    plt.title(f"Category: {category_name}")

    # 关闭坐标轴
    plt.axis('off')

    # 显示图片
    plt.show()

    # 打印一些信息
    print(f"Image shape: {image_origin.shape}")
    print(f"Number of boxes: {len(box_poses)}")
    print(f"Category ID: {category_id}")
    print(f"Category Name: {category_name}")
