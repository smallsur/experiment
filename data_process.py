from tkinter import Image
import numpy as np
import pandas as pd
import json
import h5py
from tqdm import tqdm


fy = h5py.File('/media/awen/D/dataset/coco_detections.hdf5','r')

# print(f.keys())




caption_train=None
caption_val=None
ins_train=None
ins_val=None
with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/captions_train2014.json','r') as f:
    caption_train = json.load(f)#image = 82783 ann = 414113


with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/captions_val2014.json','r') as f:
    caption_val = json.load(f)#image = 40504 ann = 202654

# with open('/media/awen/D/dataset/coco_2014/annotations/instances_train2014.json','r') as f:
#     ins_train = json.load(f)#82783 604907

# with open('/media/awen/D/dataset/coco_2014/annotations/instances_val2014.json','r') as f:
#     ins_val = json.load(f)#40504 291875

examples = {}#_features,_boxes,_cls_prob
val_examples = {}
test_examples ={}
imageid_to_id = {}
imageid_to_id_box = {}

for data in [caption_train,caption_val]:

    for item in data.get('annotations'):
        examples[item['id']] = {'image_id':item['image_id'],'caption':item['caption']}
        if item['image_id'] not in imageid_to_id:
            imageid_to_id[item['image_id']]=[]
        imageid_to_id[item['image_id']].append(item['id'])


    for item in data.get('images'):
        ids = imageid_to_id[item['id']]
        for i in ids:
            examples[i]['file_name']=item['file_name']
    
print("Switching")


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
with tqdm(desc='process', unit='it', total=len(imageid_to_id.keys())) as pbar:
    for it, image_id in enumerate(imageid_to_id.keys()):#_features,_boxes,_cls_prob
        for i in ids:
            data = fy['%d_boxes'%image_id][()]
            examples[i]['box'] = data.tolist()
            data = fy['%d_cls_prob'%image_id][()]
            examples[i]['label'] = np.argmax(data,-1).tolist()
        pbar.update()

with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/convert.json','w+') as f:
    json.dump(examples,f)

with open('/media/awen/D/dataset/rstnet/Datasets/m2_annotations/imageid_to_id.json','w+') as f:
    json.dump(imageid_to_id, f)

# with open('/media/awen/D/dataset/coco_2014/annotations/imageid_to_id_box.json','w+') as f:
#     json.dump(imageid_to_id_box, f)











