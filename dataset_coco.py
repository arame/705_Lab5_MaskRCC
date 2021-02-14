import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import pycocotools
from pycocotools.coco import COCO
import skimage.io as io


# dataset interface takes the ids of the COCO classes
class COCOData(data.Dataset):

     def __init__(self, **kwargs):

        self.stage = kwargs['stage']
        self.coco_classes_ids = kwargs['coco_classes_idx']
        self.adjusted_idx = kwargs['adjusted_classes_idx']
        self.coco_interface = kwargs['coco_interface']
        # this returns the list of image objects, equal to the number of images of the relevant class(es)
        self.datalist = kwargs['datalist'] 
        # load the list of the image
        self.img_data = self.coco_interface.loadImgs(self.datalist)

     # this method normalizes the image and converts it to Pytorch tensor
     # Here we use pytorch transforms functionality, and Compose them together,
     def transform(self, img):
         # these mean values are for RGB!!
         t_ = transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.ToTensor(),
                             #transforms.Normalize(mean=[0.485, 0.457, 0.407],
                             #                     std=[1,1,1])
                             ])

  
         img = t_(img)
         # need this for the input in the model
         # returns image tensor (CxHxW)
         return img

     # downloadthe image 
     # return rgb image
     def load_img(self, idx):      
       im = np.array(io.imread(self.img_data[idx]['coco_url']))
       im = self.transform(im)
       return im

     def load_label(self, idx): 
         # extract the id of the image, get the annotation ids
         im_id = self.img_data[idx]['id']
         annIds = self.coco_interface.getAnnIds(imgIds = im_id, catIds = self.coco_classes_ids, iscrowd=None)  
         # get the annotations 
         anns = self.coco_interface.loadAnns(annIds)         
         boxes = []
         ids = []
         masks = []
         # loop through all objects in the image
         # append id, bbox, extract mask and append it too
         for a in anns:
             adjusted_id = self.adjusted_idx[a['category_id']] 
             ids.append(adjusted_id)             
             box_coco = a['bbox']
             box = [box_coco[0], box_coco[1], box_coco[0]+box_coco[2], box_coco[1]+box_coco[3]]
             boxes.append(box)             
             # MS COCO stores masks in rle format, pass the whole annotation to the rle
             mask_object = self.coco_interface.annToMask(a)
             masks.append(mask_object)
         # Careful with the data types!!!!
         # Also careful with the variable names!!!
         # If you accidentally use the same name for the object labels and the labs (output of the method) 
         # you get an infinite recursion
         boxes = torch.as_tensor(boxes, dtype = torch.float)
         ids = torch.tensor(ids, dtype=torch.int64)
         masks = torch.tensor(masks, dtype=torch.uint8)
         labs = {}
         labs['boxes'] = boxes
         labs['labels'] = ids
         labs['masks'] = masks
         return labs

     # number of images
     def __len__(self):
         return len(self.datalist)


     # return image + mask 
     def __getitem__(self, idx):
         X = self.load_img(idx)
         y = self.load_label(idx) 
         return X,y
