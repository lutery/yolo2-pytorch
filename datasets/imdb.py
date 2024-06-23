import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import cfgs.config as cfg
import cv2


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


class ImageDataset(object):
    def __init__(self, name, datadir, batch_size, im_processor,
                 processes=3, shuffle=True, dst_size=None):
        self._name = name
        self._data_dir = datadir
        self._batch_size = batch_size
        self.dst_size = dst_size

        self._epoch = -1
        self._num_classes = 0
        self._classes = []

        # load by self.load_dataset()
        self._image_indexes = []
        self._image_names = [] # 存储所有的图片名字，这个方法需要由子类实现
        self._annotations = []
        # Use this dict for storing dataset specific config options
        self.config = {}

        # Pool
        self._shuffle = shuffle
        # 进程池数
        self._pool_processes = processes
        # 创建进程池
        self.pool = Pool(self._pool_processes)
        self.gen = None # todo 作用
        self._im_processor = im_processor

    def next_batch(self, size_index):
        '''
        返回一个batch的训练数据

        param size_index: todo 
        return: todo dict, 包含images, gt_boxes, gt_classes, dontcare, origin_im
        '''

        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [],
                 'dontcare': [], 'origin_im': []}
        i = 0
        if self.gen is None:
            indexes = np.arange(len(self.image_names), dtype=np.int)
            if self._shuffle:
                np.random.shuffle(indexes)
            # 采用进程池的操作，将数据放入进程池中，使用imap分别进行处理
            self.gen = self.pool.imap(partial(self._im_processor,
                                              size_index=None),
                                      ([self.image_names[i],
                                        self.get_annotation(i),
                                        self.dst_size] for i in indexes),
                                      chunksize=self.batch_size)
            self._epoch += 1
            print(('epoch {} start...'.format(self._epoch)))

        # 开始获取数据
        while i < self.batch_size:
            try:
                # 从_im_processor中获取处理后的图片数据
                # 貌似图片还没有做归一化
                images, gt_boxes, classes, dontcare, origin_im = next(self.gen)

                # multi-scale 
                # 这里是从一个预先设置的图片尺寸表中获取想要变换的
                # 图片尺寸，应该也是一种图片增强操作，实现多种尺度图片的训练
                w, h = cfg.multi_scale_inp_size[size_index]
                # 根据获取的图片尺寸对预测框和图片进行尺寸变化
                gt_boxes = np.asarray(gt_boxes, dtype=np.float)
                if len(gt_boxes) > 0:
                    gt_boxes[:, 0::2] *= float(w) / images.shape[1]
                    gt_boxes[:, 1::2] *= float(h) / images.shape[0]
                images = cv2.resize(images, (w, h))

                # batch存储着：处理后的图片、预测框、预测框的类别、不关心的区域、原始图片
                # 不关心区域看代码应该是空的
                batch['images'].append(images)
                batch['gt_boxes'].append(gt_boxes)
                batch['gt_classes'].append(classes)
                batch['dontcare'].append(dontcare)
                batch['origin_im'].append(origin_im)
                i += 1
            except (StopIteration,):
                # 抛出异常，说明数据已经处理完了
                # todo 看起来pool imap next再处理完成后会抛出StopIteration异常
                indexes = np.arange(len(self.image_names), dtype=np.int)
                if self._shuffle:
                    np.random.shuffle(indexes)
                self.gen = self.pool.imap(partial(self._im_processor,
                                                  size_index=None),
                                          ([self.image_names[i],
                                            self.get_annotation(i),
                                            self.dst_size] for i in indexes),
                                          chunksize=self.batch_size)
                self._epoch += 1
                print(('epoch {} start...'.format(self._epoch)))
        batch['images'] = np.asarray(batch['images'])
        return batch

    def close(self):
        self.pool.terminate()
        self.pool.join()
        self.gen = None

    def load_dataset(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def get_annotation(self, i):
        if self.annotations is None:
            return None
        return self.annotations[i]

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_names(self):
        '''
        返回所有图片的名字
        '''

        return self._image_names

    @property
    def image_indexes(self):
        return self._image_indexes

    @property
    def annotations(self):
        return self._annotations

    @property
    def cache_path(self):
        cache_path = os.path.join(self._data_dir, 'cache')
        mkdir(cache_path)
        return cache_path

    @property
    def num_images(self):
        '''
        图片的数量
        '''
        return len(self.image_names)

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        '''
        每次迭代的batch大小
        '''
        return self._batch_size

    @property
    def batch_per_epoch(self):
        '''
        TODO 这里应该是返回需要迭代多少次可以完成数据集所有数据的迭代的次数
        '''
        return self.num_images // self.batch_size
