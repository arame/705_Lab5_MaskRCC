import torch
from PIL import Image as PILImage
import os,sys,re
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

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

print(id_to_chars, id_to_chars[45])


sign_detector_weights = torch.load("maskrcnn_model_icdar2015_fst_masks.pth", map_location="cpu")

inference_args = {'num_classes':len(chars_list)}
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, **inference_args)
maskrcnn_model.load_state_dict(sign_detector_weights)
maskrcnn_model.eval()

def icdar_mask_rcnn_inference(im_input, problem, model):
    threshold = 0.75
    im = PILImage.open(im_input)
    # get rid of alpha channel
    img = np.array(im)
    if img.shape[2]>3:
       img=img[:,:,:3]
    t_ = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.457, 0.407],
                    #                     std=[1,1,1])
                             ])

    img = t_(img)
    out = model([img])
    # scores + bounding boxes + labels + masks
    scores = out[0]['scores']
    bboxes = out[0]['boxes']
    classes = out[0]['labels']
    # masks: unmolded, size of the whole image
    if problem == "mask":
       mask = out[0]['masks']
       color_array = np.zeros([mask[0].shape[1], mask[0].shape[2],3], dtype=np.uint8)

    best_scores = scores[scores>threshold]
    best_idx = np.where(scores>threshold)
    best_bboxes = bboxes[best_idx]
    best_classes = classes[best_idx]
    # for the objects with scores>threshold: 
    if len(best_idx)>0:
       bgr_img = cv2.imread(im_input)
       #plt.imshow(im)
       ax = plt.gca()
       plt.margins(0,0)
       # plot masks
       # Do not use pytorch tensors 
       # before plotting, convert to numpy
       for idx, b in enumerate(best_bboxes):
           found = mask[idx][0].detach().clone().cpu().numpy()
           col = 255*np.random.rand(3)
           color_array[found>0.5] = col
           lw = best_scores[idx]*2 
           rect = Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], linewidth=lw, edgecolor='b', facecolor='none')
           ax.text(b[0],b[1], id_to_chars[best_classes[idx].item()], fontsize=20, color='red', fontweight='bold')
           ax.add_patch(rect)

       added_image = cv2.addWeighted(bgr_img, 0.5, color_array, 0.75, 0)       
       plt.imshow(added_image)
       plt.show()
       #plt.imsave("sign1_bboxes.jpg", added_image)

icdar_mask_rcnn_inference("sign1.jpg", "mask", maskrcnn_model) 
