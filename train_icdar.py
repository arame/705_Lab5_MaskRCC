###############################################   LAB 5   ##############################################
################### Script written by Dr Alex Ter-Sarkisov@City, University of London, 2020 ############
##################### DEEP LEARNING CLASSIFICATION, MSC IN ARTIFICIAL INTELLIGENCE #####################
########################################################################################################
import time
import torch
import torchvision
import numpy as np
import os,sys
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils import data
import torch.utils as utils
from torchvision import transforms
import dataset_icdar
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

device = torch.device('cpu')


if torch.cuda.is_available():
   device = torch.device('cuda')

ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def icdar_mask_rcnn_inference(im, problem, model):
    threshold = 0.75
    im = PILImage.open(im)
    img = np.array(im)
    t_ = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.457, 0.407],
                                         std=[1,1,1])
                     ])

    img = t_(img)
    print(img.size())
    #img = img.to(device)
    out = model([img])
    # scores + bounding boxes + labels + masks
    scores = out[0]['scores']
    bboxes = out[0]['boxes']
    classes = out[0]['labels']
    print(out)
    # masks: unmolded, size of the whole image
    if problem == "mask":
       mask = out[0]['masks']
       print(mask[0].shape)
       color_array = np.zeros([mask[0].shape[1], mask[0].shape[2],3], dtype=np.uint8)
       print('color', color_array.shape)

    scores = scores.detach().clone().cpu().numpy()
    best_idx = np.where(scores>threshold)
    best_bboxes = bboxes[best_idx]
    best_classes = bboxes[best_idx]
    # for the objects with scores>threshold: 
    # add bbox and mask 
    if len(best_idx)>0:
       bgr_img = cv2.imread("dogcat1.jpg")
       #plt.imshow(im)
       ax = plt.gca()
       # plot masks
       # Do not use pytorch tensors 
       # before plotting, convert to numpy
       for id, b in enumerate(best_bboxes):
           #if classes[id] == 17:
           found = mask[id][0].detach().clone().cpu().numpy()
           color_array[found>0.5] = [255,0,0]
           #elif classes[id] ==18:
           #   found = mask[id][0].detach().clone().cpu().numpy()
           #   color_array[found>0.5] = [0,255,0]
           rect = Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='r', facecolor='none')
           ax.add_patch(rect)

       added_image = cv2.addWeighted(bgr_img, 0.5, color_array, 0.5, 0)       
       plt.imsave("sign1_bboxes.jpg", bgr_img)


chars_list=[]
for x in ascii_letters:
    chars_list.append(x)

for i in range(10):
    chars_list.append(str(i))
chars_list.insert(0, '__bgr__')

# both dicts:chars to id and id to chars
chars_to_id = {}
id_to_chars = {}
for id, c in enumerate(chars_list):
    chars_to_id[c] = id
    id_to_chars[id] = c

print(chars_to_id)
mask_ids = chars_to_id
# Load ICDAR dataset
# defien the problem
problem = 'mask'
if problem == 'mask' or problem == 'char':
   gt_dir = 'gt3'
elif problem == 'word':
   gt_dir = 'gt1'

mask_dir = 'gt2'
# parameters for the dataset
# if only signs are detected, gt_dir is gt1
# for characters gt_dir is gt3
# for masks gt2 and gt3
dataset_pars_sign = {'stage':'train', 'gt':os.path.join('icdar', gt_dir),'data':'icdar/img','problem':problem,'img_max_size':[256,256], 'list_of_chars':chars_list, 'char_dict':chars_to_id, 'mask_gt': os.path.join('icdar',mask_dir)}
datapoint_sign = dataset_icdar.ICDARData(**dataset_pars_sign)
dataloader_pars_sign = {'shuffle':True, 'batch_size':1}
dataloader_pars = data.DataLoader(datapoint_sign, **dataloader_pars_sign)
#dataset_pars_valid = {'stage':'validation', 'gt_dir':'../icdar2015_fst/gt2','img_dir':'../icdar2015_fst/img'}
#datapoint_validation = dataset.DatasetICDAT(**dataset_pars_valid)
######################################################
if problem == 'char' or problem == 'mask':
   maskrcnn_args = {'num_classes':len(chars_list)}
elif problem == 'word':
   maskrcnn_args = {'num_classes':2}
####################### EVAL OF THE PRETRAINED MODEL #############################
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, **maskrcnn_args)
print(maskrcnn_model)
pretrained_weights = torch.load(os.path.join("pretrained_weights","maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"))
# copy only backbone weights
for _n, _par in maskrcnn_model.named_parameters():
    if 'backbone' in _n:
       print(_n)
       _par.requires_grad = False
       _par.copy_(pretrained_weights[_n])
       _par.requires_grad = True

maskrcnn_model.train()

if device == torch.device('cuda'):
   maskrcnn_model = maskrcnn_model.to(device)

optimizer_pars = {'lr':1e-5, 'weight_decay':1e-3}
optimizer = torch.optim.Adam(list(maskrcnn_model.parameters()),**optimizer_pars)


total_epochs = 5

start_time = time.time()

for e in range(total_epochs):
    epoch_loss = 0
    total_img = 0
    for b in dataloader_pars:
        optimizer.zero_grad()
        total_img += 1
        X,y = b
        if device==torch.device('cuda'):
            X, y['labels'], y['boxes'] = X.to(device), y['labels'].to(device), y['boxes'].to(device)
            if problem == 'mask':
               y['masks'] = y['masks'].to(device)
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
        if problem == 'mask':
           lab['masks'] = y['masks'].squeeze_(0)
        targets.append(lab)
        # avoid empty objects
        if len(targets)>0:
           loss = maskrcnn_model(images, targets)
           total_loss = 0
           for k in loss.keys():
               total_loss += loss[k]
           epoch_loss += total_loss.clone().detach().cpu().numpy()
           total_loss.backward()
           optimizer.step()
    epoch_loss = epoch_loss/total_img
    print("Loss in epoch {0:d} = {1:.3f}".format(e, epoch_loss))

end_time = time.time()

print("Training took {0:.1f}".format(end_time-start_time))
# inference + save the model
maskrcnn_model = maskrcnn_model.to(torch.device('cpu'))

maskrcnn_model.eval()

if problem == 'mask':
   torch.save(maskrcnn_model.state_dict(), "maskrcnn_model_icdar2015_fst_masks.pth")
elif problem == 'char':
   torch.save(maskrcnn_model.state_dict(), "maskrcnn_model_icdar2015_fst_char.pth")
elif problem == 'word':
   torch.save(maskrcnn_model.state_dict(), "maskrcnn_model_icdar2015_fst_word.pth")


icdar_mask_rcnn_inference("sign1.jpg", problem, maskrcnn_model)

