
import yaml

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.utils.data as data
from fmutils import fmutils as fmu
from empatches import EMPatches
from tabulate import tabulate
# from PIL import Image
import cv2
import numpy as np
import os, random, time
import matplotlib.cm as cm
import torch
from data.augmenters import data_augmenter
from data.utils import std_norm
from pathlib import Path

emp = EMPatches()

class GEN_DATA_LISTS():
    
    def __init__(self, root_dir, sub_dirname):
        '''
        Parameters
        ----------
        root_dir : TYPE
            root directory containing [train, test, val] folders.
        sub_dirname : TYPE
            sub directories inside the main split (train, test, val) folders
        get_lables_from : TYPE
            where to get the label from either from dir_name of file_name.


        '''
        self.root_dir = root_dir
        self.sub_dirname = sub_dirname
        self.splits = ['train', 'test']
        
    def get_splits(self):
        
        print('Directories loadded:')
        self.split_files = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[0]))
            self.split_files.append(os.path.join(self.root_dir, split, self.sub_dirname[0]))
        print('\n')
        
        self.split_depths = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[1]))
            self.split_depths.append(os.path.join(self.root_dir, split, self.sub_dirname[1]))
        print('\n')
        
        self.split_f_line = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[2]))
            self.split_f_line.append(os.path.join(self.root_dir, split, self.sub_dirname[2]))
        print('\n')
        
        self.split_f_seg = []
        for split in self.splits:
            print(os.path.join(self.root_dir, split, self.sub_dirname[3]))
            self.split_f_seg.append(os.path.join(self.root_dir, split, self.sub_dirname[3]))
        print('\n')
            
        
        self.train_f = fmu.get_all_files(self.split_files[0])
        self.test_f = fmu.get_all_files(self.split_files[1])

        self.train_d = fmu.get_all_files(self.split_depths[0])
        self.test_d = fmu.get_all_files(self.split_depths[1])
        
        self.train_fl = fmu.get_all_files(self.split_f_line[0])
        self.test_fl = fmu.get_all_files(self.split_f_line[1])
        
        self.train_fs = fmu.get_all_files(self.split_f_seg[0])
        self.test_fs = fmu.get_all_files(self.split_f_seg[1])
        
        train, test = [self.train_f, self.train_d, self.train_fl, self.train_fs], [self.test_f, self.test_d, self.test_fl, self.test_fs]
        
        return train, test
    
    def get_classes(self):
        
        cls_names = []
        for i in range(len(self.train_f)):
            cls_names.append(fmu.get_basename(self.train_f[i]).split('_')[1])
        classes = sorted(list(set(cls_names)), key=fmu.numericalSort)
        return classes
    
    def get_filecounts(self):
        print('\n')
        result = np.concatenate((np.asarray(['train', 'test']).reshape(-1,1),
                                np.asarray([len(self.train_f), len(self.test_f)]).reshape(-1,1),
                                np.asarray([len(self.train_d), len(self.test_d)]).reshape(-1,1))
                                , 1)
        print(tabulate(np.ndarray.tolist(result), headers = ["Split", "Images", "Labels"], tablefmt="github"))
        return None

class Furrow(data.Dataset):
    def __init__(self, img_paths, depth_paths, furrow_line, furrow_seg, img_height, img_width, augment_data=False, normalize=False):
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.furrow_line = furrow_line
        self.furrow_seg = furrow_seg
        self.img_height = img_height
        self.img_width = img_width
        self.augment_data = augment_data
        self.normalize = normalize
        self.my_epoch = 1
        self.idx = 0

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        data_sample = {}
        # print(self.img_paths[index])
        # read RGB image
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        # read depth map
        depth = cv2.imread(self.depth_paths[index], -1) # this is uint16 image
        depth = cv2.resize(depth, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)#.astype(np.uint8)
        max_threshold = min(10000, np.max(depth)) # greater then 10 meters are useless
        # print(np.max(depth), np.min(depth))
        # read furrow line
        f_line = (cv2.imread(self.furrow_line[index], 0) / 255.0).astype(np.uint8)
        f_line = cv2.resize(f_line, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        f_line = cv2.dilate(f_line, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)), iterations = 1)
        # read furrow segments
        f_seg = (cv2.imread(self.furrow_seg[index], 0) / 255.0).astype(np.uint8)
        f_seg = cv2.resize(f_seg, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        
        if self.augment_data:
            img, depth, f_line, f_seg, a, b = data_augmenter(img, depth, f_line, f_seg, self.my_epoch)
        else:
            a, b = 0, 0 # no augmentation

        if self.normalize:
            img = std_norm(img)
            depth = np.clip(depth / max_threshold, 0, 1)
            depth = cm.jet(depth)[...,0:3]
            # depth = depth * 2 - 1 Normalization between -1 and 1
        
        assert len(np.unique(f_line)) <= config['num_classes'], f'A total of {len(np.unique(f_line))} labels found in {self.furrow_line[index]}.'
        
        self.idx += 1
        if len(self.img_paths) / self.idx < 2: # one iteration over data finished b/c (x+1)/x < 2.
            self.my_epoch += 1
            self.idx = 0 # reset

        data_sample['img'] = img
        data_sample['depth'] = depth
        data_sample['f_line'] = f_line
        data_sample['f_seg'] = f_seg
        data_sample['geo_augs'] = a
        data_sample['noise_augs'] = b
        
        return data_sample 
    
    def return_epoch(self):
        return self.my_epoch
