import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt

# dataset for sign detection and char detection
class ICDARData(data.Dataset):

     def __init__(self, **kwargs):           
         self.stage = kwargs['stage']
         self.problem = kwargs['problem']
         if self.problem == 'char' or self.problem == 'mask':
            self._chars = kwargs['list_of_chars']
            self._char_dict = kwargs['char_dict']
            if self.problem == 'mask':
               self.dir_masks = kwargs['mask_gt']
                 
         # this returns the path to data dir
         self.data = kwargs['data']         
         # this returns the path to 
         self.gt = kwargs['gt']
         self.img_max_size = kwargs['img_max_size']
         self.sorted_data = sorted(os.listdir(self.data))
         
     # this method normalizes the image and converts it to Pytorch tensor
     # Here we use pytorch transforms functionality, and Compose them together,
     def transform_img(self, img, img_max_size):
         h,w,c = img.shape
         h_,w_ = img_max_size[0], img_max_size[1]
         img_size = tuple((h_,w_))
         # Faster R-CNN does the normalization
         t_ = transforms.Compose([
                             transforms.ToPILImage(),
                             #transforms.Resize(img_size),
                             transforms.ToTensor(),
                             ])
         img = t_(img)
         return img


     def load_img(self, idx):
         im = np.array(PILImage.open(os.path.join(self.data, self.sorted_data[idx])))
         im = self.transform_img(im, self.img_max_size)
         return im

     # bounding box for chars
     # or whole words
     def load_label_icdar(self, idx, type):
         self.list_of_bboxes = []
         self.list_of_chars = []
         self.labels =  []
         if self.problem == "mask":
            self.list_of_masks = []
         # load bbox for the whole word
         # 
         if type == "word":  
            fname = os.path.join(self.gt, 'gt_' + self.sorted_data[idx].split('.')[0]+'.txt')
            with open(fname, "r") as f:
                 ln = [x.rstrip() for x in f.readlines()]
            for l in ln:
                # only one class
                obs = l.split()
                self.labels.append(1) 
                # minx, miny, maxx, maxy
                bbox = [int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])]
                self.list_of_bboxes.append(bbox)

            self.lab = {}
            self.list_of_bboxes = torch.tensor(self.list_of_bboxes, dtype=torch.float)
            self.labels = torch.tensor(self.labels, dtype=torch.uint8)
            self.lab['labels'] = self.labels
            self.lab['boxes'] = self.list_of_bboxes
         # load bbox for each char
         # a total of 2x26+1
         # load masks
         # label is the same as the label of char
         elif type == "char" or type == "mask":
              fname = os.path.join(self.gt, self.sorted_data[idx].split('.')[0]+'_GT.txt')
              # this line returns the list of observation: rgb data+centroids of bbox + coords of bboxes +chars
              with open(fname, "r") as f: 
                 char_labs = [x.rstrip() for x in f.readlines()]
                 # get rid of empty spaces
                 char_labs = [x for x in char_labs if len(x)>0] 
                 # get every observation
                 for obs in char_labs:                   
                     d = obs.split()
                     char = re.sub('\W+','',d[-1])
                     if char in self._chars:
                        # get the character
                        self.list_of_chars.append(self._char_dict[char])
                        # extract the bbox 
                        bbox = [int(d[5]), int(d[6]), int(d[7]), int(d[8])]
                        self.list_of_bboxes.append(bbox)
                        # if mask prediction, add a mask of the label
                        if type == "mask":
                           mask_name = self.sorted_data[idx].split('.')[0]+'_GT.bmp'
                           mask = PILImage.open(os.path.join(self.dir_masks,mask_name))
                           mask = np.array(mask)
                           aux_array = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.uint8)
                           # crop the bounding box from the image mask into its map
                           # convert to integers, to crop the objects
                           # get rid of teh background, and keep just the first dimension
                           # identify all positive pixels and convert them to object class
                           crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                           crop[crop==[255,255,255]] = 0
                           crop = crop[:,:,0]
                           crop[crop>0]=1
                           aux_array[bbox[1]:bbox[3], bbox[0]:bbox[2]] = crop
                           #some masks come out as all 0s, ignore them
                           if np.sum(aux_array)>0:
                              # add mask to the list
                              self.list_of_masks.append(aux_array)
                           else:
                              # get rid of label and bbox
                              del self.list_of_bboxes[-1]
                              del self.list_of_chars[-1]

              self.labels = torch.tensor(self.list_of_chars, dtype=torch.int64)
              self.list_of_bboxes = torch.tensor(self.list_of_bboxes, dtype=torch.float)
              # output labels, bboxes and masks
              self.lab={}
              self.lab['labels'] = self.labels
              self.lab['boxes'] = self.list_of_bboxes
              # add masks
              if type == "mask":
                self.lab['masks'] = torch.as_tensor(self.list_of_masks, dtype=torch.uint8)

         return self.lab 

     #'magic' method: size of the dataset
     def __len__(self):
         return len(os.listdir(self.data))

     # return one datapoint
     def __getitem__(self,idx):

         X = self.load_img(idx)
         y = self.load_label_icdar(idx, self.problem)
         return X,y


