import torch
from torch.utils.data import Dataset

class TK100Dataset(Dataset):
    def __init__(self, data_info_file_path, category_path, transform=None, dataset_path=r"F:\Projects\datasets\oc\TK100\data"):
        super().__init__()

        self.data_info_file_path = data_info_file_path
        self.category_path = category_path
        self.transform = transform
        self.box_names = []
        self.box_poses = []
        self.box_labels = []
        self.num_boxes = []
        self.dataset_main_path = dataset_path
        self.__load_oc_data()
        self.__load_oc_category()

    
    def __load_oc_data(self):
        with open(self.data_info_file_path) as f:
            lines = f.readlines()

        for line in lines:
            box_infos = line.strip().split(' ')
            self.box_names.append(box_infos)
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
            self.box_poses.extend(box_pos)
            self.box_labels.extend(box_label)
        self.num_boxes = len(self.box_poses)


    def __load_oc_category(self):
        with open(self.category_path) as f:
            categorys = f.readlines()
        self.category_2_id = {category.strip():idx for idx, category in enumerate(categorys)}
        self.id_2_category = {idx:category.strip() for idx, category in enumerate(categorys)}
        

    def __getitem__(self, index):
        
    
    def __len__(self):
        return self.num_boxes
    

if __name__ == "__main__":
    data_info_file_path = r'M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100.txt'
    category_path = r'M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100-catetory.txt'
    tk100_dataset = TK100Dataset(data_info_file_path, category_path)
    print(tk100_dataset.category_2_id)
    print(tk100_dataset.id_2_category)
    print(tk100_dataset.num_boxes)
    print(tk100_dataset.box_poses)
    print(tk100_dataset.box_labels)
    print(tk100_dataset.box_names)