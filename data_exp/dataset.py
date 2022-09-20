from ast import Import
from distutils.spawn import spawn
import os
from re import S
import numpy as np
from torch.utils.data import Dataset,DataLoader
import h5py
from tqdm import tqdm
import itertools
import collections
import torch
import pickle
# import sys
# sys.path.append('/media/awen/D/experiment')

# import data

from .utils import load_json
from .example import Example
from .field import ImageDetectionsField,TextField,BoxField,RawField

class FieldDataset(Dataset):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)
    
    
    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))
        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    
    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)



class ValueDataset(FieldDataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(FieldDataset):
    def __init__(self, examples, fields, key_fields):

        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []
        with tqdm(desc='create dict',unit='i',total=len(examples)) as pbar:
            for i, e in enumerate(examples):
                key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
                value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
                if key_example not in key_dict:
                    key_dict[key_example] = len(key_examples)
                    key_examples.append(key_example)

                value_examples.append(value_example)
                dictionary[key_dict[key_example]].append(i)
                pbar.update()

        self.key_dataset = FieldDataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return *key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)
        


class COCO(object):
    def __init__(self, img_path, ann_path,fields):

        self.img_path = img_path
        self.ann_path = ann_path

        # assert self.img_path,self.ann_path

        ann_dict = load_json(os.path.join(self.ann_path, 'convert.json'))
        # imageid_to_id = load_json(os.path.join(self.ann_path, 'imageid_to_id.json'))

        ids={}
        t1 = np.load(os.path.join(ann_path, 'coco_train_ids.npy'))
        t2 = np.load(os.path.join(ann_path, 'coco_restval_ids.npy'))
        t3 = []
        for t in [t1,t2]:
            for i in t:
                t3.append(i)
        ids['train'] = t3#566435
        ids['val'] = np.load(os.path.join(ann_path, 'coco_dev_ids.npy'))#25000
        ids['test'] = np.load(os.path.join(ann_path, 'coco_test_ids.npy'))#25000

        # fy = h5py.File(dect_path,'r')

        self.train_examples, self.val_examples, self.test_examples = self.get_samples(ann_dict,ids)#616767
        self.fields = fields
        self.fields_evalue = {k : v for k,v in self.fields.items() if k in ['image','box']}
        self.fields_evalue['text'] = RawField()
        self.datasets = {}
        self.evalue_datasets = {}
        

    @classmethod
    def get_samples(cls, ann_dict,ids):

        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            with tqdm(desc='split %s'%split,unit='i',total=len(ids[split])) as pbar:
                id_ = ids[split]
                for i,id in enumerate(id_):
                    annotation = ann_dict[str(id)]['caption']
                    filename = ann_dict[str(id)]['file_name']
                    image_id = ann_dict[str(id)]['image_id']

                    example = Example.fromdict({'text': annotation, 'image': filename, 'box':image_id})

                    if split == 'train':
                        train_samples.append(example)
                    elif split == 'val':
                        val_samples.append(example)
                    elif split == 'test':
                        test_samples.append(example)
                    pbar.update()

        return train_samples,val_samples,test_samples

    def get_dataset(self):

        if 'train' not in self.datasets:
            self.datasets['train'] = FieldDataset(self.train_examples,self.fields)
            self.datasets['val'] = FieldDataset(self.val_examples,self.fields)
            self.datasets['test'] = FieldDataset(self.test_examples,self.fields)

        return self.datasets

    def get_evalue_dataset(self):
        if 'e_train' not in self.evalue_datasets:
            self.evalue_datasets['e_train'] = DictionaryDataset(self.train_examples,self.fields_evalue,key_fields=('image','box'))
            self.evalue_datasets['e_val'] = DictionaryDataset(self.val_examples,self.fields_evalue,key_fields=('image','box'))
            self.evalue_datasets['e_test'] = DictionaryDataset(self.test_examples,self.fields_evalue,key_fields=('image','box'))
        return self.evalue_datasets

    # def get_dataloader(self,dataset_,key,args=None):
    #     if key in ['train','e_train']:
    #         return DataLoader(dataset=dataset_,collate_fn=dataset_.collate_fn(),batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
        
    #     return DataLoader(dataset=dataset_,collate_fn=dataset_.collate_fn(),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)



if __name__ == '__main__':

    tf = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    imf = ImageDetectionsField(detections_path='/media/awen/D/dataset/rstnet/Datasets/X101-features/coco_X101_grid.hdf5',max_detections=49)
    bf = BoxField()
    f = { 'image': imf, 'text': RawField(),'box':bf}

    tf.vocab = pickle.load(open('/media/awen/D/dataset/rstnet/vocab.pkl', 'rb'))

    coco = COCO('','/media/awen/D/dataset/rstnet/Datasets/m2_annotations', fields=f)
    
    datasets = coco.get_dataset()
    evalue_datasets = coco.get_evalue_dataset()

    loader = coco.get_dataloader(datasets['test'],'test')

    for i,data in enumerate(loader):
        print(data)






    


    



        
        




        