from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
import glob
from PIL import Image,ImageOps,ImageEnhance
import matplotlib.pyplot as plt
import torch
import pandas as pd 
import ast 
from utils import *
from random import sample

class Channels(Dataset):
    def __init__(self, path = None,ext = 'txt',labels_path = None,sampling = None):       
        self.path = path
        self.ext =ext
        self.labels_path = labels_path
        
        self.img_list = os.listdir(path)
        if sampling:
            self.img_list = sample(self.img_list,sampling)
        if self.ext != 'txt' and self.ext != 'grdecl': # png,jpg
            self.transform1 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        else:
            self.transform1 = None
            
    def __len__(self):
        return len(self.img_list)
    
    def split_string_at_last_symbol(self,input_string,symbol = '/'):
        last_slash_index = input_string.rfind(symbol)
        if last_slash_index == -1:
            return input_string, ''
        else:
            first_part = input_string[:last_slash_index]
            second_part = input_string[last_slash_index + 1:]
        return first_part, second_part

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        full_img_path = os.path.join(self.path,img_name)
        
        if self.ext == 'txt': 
            x = np.loadtxt(full_img_path) #normalized
            img = torch.from_numpy(x).unsqueeze(0).float()
        elif self.ext == 'png':
            img =Image.open(full_img_path)
        elif self.ext == 'grdecl':
            img =read_gcl(full_img_path) 
            img = normalize_by_replace(img)
            img = torch.from_numpy(img).unsqueeze(0).float()
          
          
        if self.transform1:
            img = self.transform1(img)   
        
        if self.labels_path == None:
            return {0: img}
        else:
            img_raw_name,ext = self.split_string_at_last_symbol(img_name,'.')
            img_label_path = os.path.join(self.labels_path,img_raw_name+'.npy')
            l = np.load(img_label_path)
            l = torch.tensor(l).float()
            return {0: img, 1: l}


def read_gcl(filepath):
    file_pointer = open(f"{filepath}", "r")
    data_list = []
    for line in file_pointer:
        line = line.strip().split(' ')

        if line[0] == '--' or line[0] == '' or line[0].find('_') > 0:
            continue
        for data_str_ in line:
            if data_str_ == '':
                continue
            elif data_str_.find('*') == -1:
                try:
                    data_list.append(int(data_str_))
                except:
                    pass # automatically excludes '/'
            else:
                run = data_str_.split('*')
                inflation = [int(run[1])] * int(run[0])
                data_list.extend(inflation)

    file_pointer.close()
    data_np = np.array(data_list)
    # print(data_np.shape)
    data_np = data_np.reshape(100, 100)
    return data_np

d = {2:-1,3:0,5:1} # maps for non-stat


def normalize_by_replace(img):
    for key, value in d.items():
        img[img==np.array(key)] = value
    return img
