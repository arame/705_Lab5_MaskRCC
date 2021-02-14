###############################################   LAB 5   ##############################################
################### Script written by Dr Alex Ter-Sarkisov@City, University of London, 2020 ############
##################### DEEP LEARNING CLASSIFICATION, MSC IN ARTIFICIAL INTELLIGENCE #####################
########################################################################################################
import time
import os, sys, re
from pycocotools.coco import COCO
import torch
import torchvision
import dataset_coco
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils import data


device = torch.device('cpu')


if torch.cuda.is_available():
   device = torch.device('cuda')

###################### load COCO interface, the input is a json file with annotations ####################
coco_interface = COCO("instances_train2017.json")
# all indices of categories in MS COCO:
all_cats = coco_interface.getCatIds()
# add background class
all_cats.insert(0,0)
print(all_cats, len(all_cats))
# get names of cateogories
all_names = coco_interface.loadCats(all_cats[1:])
#print(all_names)
# choose the categories you want to work with
# VERY CAREFUL WITH THIS LIST! SOME CLASSES ARE MISSING, TO TRAIN THE MODEL
# YOU NEED TO ADJUST THE CLASS ID!!!
selected_class_ids = coco_interface.getCatIds(catNms=['person'])
adjusted_class_ids = {}
for id, cl in enumerate(all_cats):
    adjusted_class_ids[cl] = id
print ("ADJUSTED CLASS IDS:")
print(adjusted_class_ids) 
###############################################
# load ids of images with this class
# Dataset class takes this list as an input and creates data objects 
im_ids = coco_interface.getImgIds(catIds=selected_class_ids)
##############################################
# selected class ids: extract class id from the annotation
coco_data_args = {'datalist':im_ids, 'coco_interface':coco_interface, 'coco_classes_idx':selected_class_ids,'stage':'train', 'adjusted_classes_idx':adjusted_class_ids}
coco_data = dataset_coco.COCOData(**coco_data_args)
coco_dataloader_args = {'batch_size':1, 'shuffle':True}
coco_dataloader = data.DataLoader(coco_data, **coco_dataloader_args)
################### MASK R-CNN MODEL ################################################
maskrcnn_args = {'num_classes':81, 'min_size':1280, 'max_size':1280}

####################### EVAL OF THE PRETRAINED MODEL #############################
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,**maskrcnn_args)
print(maskrcnn_model)
pretrained_weights = torch.load(os.path.join('pretrained_weights', 'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'))
# copy only backbone weights
for _n, _par in maskrcnn_model.named_parameters():
    if 'backbone' in _n:
       print(_n)
       _par.requires_grad = False
       _par.copy_(pretrained_weights[_n])
       _par.requires_grad = True


if device == torch.device('cuda'):
   maskrcnn_model = maskrcnn_model.to(device)

maskrcnn_model.train()

maskrcnn_optimizer_pars = {'lr':1e-5}
maskrcnn_optimizer = optim.Adam(list(maskrcnn_model.parameters()),**maskrcnn_optimizer_pars)

total_epochs = 1

start_time = time.time()

for e in range(total_epochs):
    epoch_loss = 0
    total_img = 0
    for id, b in enumerate(coco_dataloader):
        maskrcnn_optimizer.zero_grad()
        total_img += 1
        X,y = b
        if device==torch.device('cuda'):
            X, y['labels'], y['boxes'], y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y['masks'].to(device)
        images = [im for im in X]
        targets = []
        lab={}
        # THIS IS IMPORTANT!!!!!
        # get rid of the first dimension (batch)
        # IF you have >1 images, make another loop
        # REPEAT: DO NOT USE BATCH DIMENSION 
        # Pytorch is sensitive to formats. Labels must be int64, bboxes float32, masks uint8
        lab['boxes'] = y['boxes'].squeeze_(0)
        lab['labels'] = y['labels'].squeeze_(0)
        lab['masks'] = y['masks'].squeeze_(0)
        print(images[0].size(), lab['boxes'].dtype, lab['labels'].dtype, lab['masks'].dtype, lab['masks'].size(), lab['boxes'].size(), lab['labels'].size())  
        targets.append(lab)
        # avoid empty objects
        print(lab)
        if len(targets)>0:
           loss = maskrcnn_model(images, targets)
           total_loss = 0
           for k in loss.keys():
               total_loss += loss[k]
           epoch_loss += total_loss.clone().detach().cpu().numpy()
           total_loss.backward()        
           maskrcnn_optimizer.step()
    epoch_loss = epoch_loss/total_img
    print("Loss in epoch {0:d} = {1:.3f}".format(e, epoch_loss))

end_time = time.time()

print("Training took {0:.1f}".format(end_time-start_time))

