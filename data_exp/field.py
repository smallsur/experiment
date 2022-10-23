# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from itertools import chain
import six
import torch
import numpy as np
import h5py
import os
import warnings
import shutil
from PIL import Image

from .vocab import Vocab
from .utils import get_tokenizer,load_json


class RawField(object):

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):

        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):

        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class BoxField(RawField):
    def __init__(self,preprocessing=None, postprocessing=None,dect_path = '/media/awen/D/dataset',sc_image_path='/media/awen/D/dataset/rstnet/Datasets/m2_annotations'):
        super(BoxField,self).__init__()

        self.dect = h5py.File(os.path.join(dect_path,'coco_detection_simple.hdf5'),'r')
        
        self.scale = load_json(os.path.join(sc_image_path,'image_to_scale.json'))

    def preprocess(self, x):
        temp = []
        # cv =[]

        # data = self.dect['%d_boxes'%x][()]
        # width = self.scale[str(x)]['width']
        # height = self.scale[str(x)]['height']
        
        # for b in data:
        #     c = [(b[2]+b[0])/(2*width),(b[3]+b[1])/(2*height),(b[2]-b[0])/width,(b[3]-b[1])/height]
        #     cv.append(c)

        # temp.append(np.array(cv,dtype=np.float32))

        # data = self.dect['%d_cls_prob'%x][()]
        # temp.append(np.argmax(data,-1).tolist())
 
        data = self.dect['%d_boxes'%x][()]
        temp.append(data)
        data = self.dect['%d_cls_prob'%x][()]
        temp.append(data)
        return {'id':x,'data':temp}

    def process(self, batch):
        batchs = []
        for ba in batch:
            boxes = default_collate(ba['data'][0])
            labels = default_collate(ba['data'][1])
            batchs.append({'id':ba['id'],'boxes':boxes,'labels':labels})

        return batchs

    def xyxy_to_xywh(self,id,boxes):

        width = self.scale[str(id)]['width']
        height = self.scale[str(id)]['height']

        x_c, y_c, w, h = boxes.unbind(-1)
        b = [((x_c - 0.5 * w) * width), ((y_c - 0.5 * h) * height),
            ((x_c + 0.5 * w) * width), ((y_c + 0.5 * h) * height)]

        return torch.stack(b, dim=-1),[width, height]
            



class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=100,
                 sort_by_prob=False, images_path=None):

        pixel_mean = [103.53, 116.28, 123.675]
        pixel_std = [57.375, 57.12, 58.395]

        self.max_detections = max_detections
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob #false 
        self.images_path = images_path

        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)


        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.transform = transforms.ToTensor()

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            # precomp_data = f['%d_features' % image_id][()]
            precomp_data = f['%d_grids' % image_id][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10,2048)

        # delta = self.max_detections - precomp_data.shape[0]

        # if delta > 0:
        #     precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        # elif delta < 0:
        #     precomp_data = precomp_data[:self.max_detections]

        return torch.from_numpy(precomp_data)


    # def preprocess(self, x):

    #     path_= None

    #     if 'train' in x:
    #         path_ = os.path.join(self.images_path,'train2014',x)
    #     else:
    #         path_ = os.path.join(self.images_path,'val2014',x)

    #     return self.transform(self._load_images(path_))

    def process(self, batch):
        # batch = [self.normalizer(x) for x in batch]
        batch, mask = generate_mask(batch)
        return {'batch': batch, 'mask': mask}
        
    # def _load_images(self, path):

    #     return Image.open(path).convert("RGB")
    
def generate_mask(tensor_list):

    if tensor_list[0].ndim == 3:
        d_model = tensor_list[0].shape[0]
        max_size = _max_by_axis([list(img.shape[-2:]) for img in tensor_list])
        batch_shape = [len(tensor_list)] + [d_model] + max_size
        b, d, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: , : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False

    else:
        raise ValueError('not supported')
    return tensor, mask

def _max_by_axis(the_list):

    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True):
        self.use_vocab = use_vocab #true
        self.init_token = init_token # bos
        self.eos_token = eos_token # eos
        self.fix_length = fix_length #None
        self.dtype = dtype #int64
        self.lower = lower #true
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation #true
        self.include_lengths = include_lengths #false
        self.batch_first = batch_first #true
        self.pad_token = pad_token#<pad>
        self.unk_token = unk_token#<unk>
        self.pad_first = pad_first #first
        self.truncate_first = truncate_first#false
        self.vocab = None
        self.vectors = vectors #none
        if nopoints: #false
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):

        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):

        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions
    



