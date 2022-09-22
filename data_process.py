from email.mime import image
from tkinter import Image
import numpy as np
import pandas as pd
import json
import h5py
import os
import re
from tqdm import tqdm
from data_exp.utils import load_json

# temp = []
# cv = []
#
# data = self.dect['%d_boxes' % x][()]
# width = self.scale[str(x)]['width']
# height = self.scale[str(x)]['height']
#
# for b in data:
#     c = [(b[2] + b[0]) / (2 * width), (b[3] + b[1]) / (2 * height), (b[2] - b[0]) / width, (b[3] - b[1]) / height]
#     cv.append(c)
#
# temp.append(np.array(cv, dtype=np.float32))
#
# data = self.dect['%d_cls_prob' % x][()]
# temp.append(np.argmax(data, -1).tolist())

fy = h5py.File('/media/awen/D/dataset/rstnet/coco_detections.hdf5', 'r')
fy_new = h5py.File('/media/awen/D/dataset/rstnet/coco_detection_simple.hdf5', 'r')
scale = load_json(os.path.join('/media/awen/D/dataset/rstnet/Datasets/m2_annotations', 'image_to_scale.json'))
print(len(fy_new.keys()))
# print(f.keys())
# with tqdm(desc='del', unit='i', total=len(fy.keys())) as pbar:
#     for i, k in enumerate(fy.keys()):
#         image_id = re.search('\d*',k).group()

#         if '_box' in k:
#             cv = []
#             data = fy[k][()]
#             width = scale[image_id]['width']
#             height = scale[image_id]['height']
#             for b in data:
#                 c = [(b[2] + b[0]) / (2 * width), (b[3] + b[1]) / (2 * height), (b[2] - b[0]) / width, (b[3] - b[1]) / height]
#                 cv.append(c)

#             fy_new[k] = np.array(cv, dtype=np.float32)

#         if '_cls_prob' in k:

#             data = fy[k][()]

#             fy_new[k] = np.argmax(data, -1).tolist()
#         pbar.update()



# caption_train=None
# caption_val=None
# ins_train=None
# ins_val=None
# with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/captions_train2014.json','r') as f:
#     caption_train = json.load(f)#image = 82783 ann = 414113


# with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/captions_val2014.json','r') as f:
#     caption_val = json.load(f)#image = 40504 ann = 202654

# with open('/media/awen/D/dataset/coco_2014/annotations/instances_train2014.json','r') as f:
#     ins_train = json.load(f)#82783 604907

# with open('/media/awen/D/dataset/coco_2014/annotations/instances_val2014.json','r') as f:
#     ins_val = json.load(f)#40504 291875

# examples = {}#_features,_boxes,_cls_prob
# val_examples = {}
# test_examples ={}
# imageid_to_id = {}
# imageid_to_id_box = {}
# image_to_scale = {}
# for data in [caption_train,caption_val]:

#     for item in data.get('images'):
#         image_id = item['id']
#         height = item['height']
#         width = item['width']
#         image_to_scale[image_id] ={'height':height, 'width':width}

# with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/image_to_scale.json','w+') as f:
#     json.dump(image_to_scale, f)

#     for item in data.get('annotations'):
#         examples[item['id']] = {'image_id':item['image_id'],'caption':item['caption']}
#         if item['image_id'] not in imageid_to_id:
#             imageid_to_id[item['image_id']]=[]
#         imageid_to_id[item['image_id']].append(item['id'])


    # for item in data.get('images'):
    #     ids = imageid_to_id[item['id']]
    #     for i in ids:
    #         examples[i]['file_name']=item['file_name']
    
# print("Switching")


# for data in [ins_train,ins_val]:
#     for item in data.get('annotations'):
#         if item['image_id'] ==536587:
#             print(item)
#         ids = imageid_to_id[item['image_id']]
#         for i in ids:
#             if 'bbox' not in examples[i]:
#                 examples[i]['bbox']=[]
#                 examples[i]['label']=[]
#             examples[i]['bbox'].append(item['bbox'])
#             examples[i]['label'].append(item['category_id'])

#         if item['image_id'] not in imageid_to_id_box:
#             imageid_to_id_box[item['image_id']]=[]
#         imageid_to_id_box[item['image_id']].append(item['id'])
# with tqdm(desc='process', unit='it', total=len(imageid_to_id.keys())) as pbar:
#     for it, image_id in enumerate(imageid_to_id.keys()):#_features,_boxes,_cls_prob
#         for i in ids:
#             data = fy['%d_boxes'%image_id][()]
#             examples[i]['box'] = data.tolist()
#             data = fy['%d_cls_prob'%image_id][()]
#             examples[i]['label'] = np.argmax(data,-1).tolist()
#         pbar.update()

# with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/convert.json','w+') as f:
#     json.dump(examples,f)

# with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/imageid_to_id.json','w+') as f:
#     json.dump(imageid_to_id, f)

# with open('/media/awen/D/dataset/coco_2014/annotations/imageid_to_id_box.json','w+') as f:
#     json.dump(imageid_to_id_box, f)











