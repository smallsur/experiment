from time import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import os
import pickle
import numpy as np
import itertools
import multiprocessing
import random
from shutil import copyfile
import warnings

from data_exp import Build_DataSet
from data_exp import TextField

from utils import Logger

import models.spatial_exp
from models.build import BuildModel
from models.spatial_exp.evalue_box import evalue_box
# from utils.box_ops import box_cxcywh_to_xyxy

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

warnings.filterwarnings("ignore")

import sys

sys.path.append(os.path.join(os.getcwd(), '..',))

import evaluation
from evaluation import PTBTokenizer, Cider


def evaluate_loss(model, dataloader, loss_fn, text_field):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections,targets, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)

                # if e == 0:#在第一个周期生成ground-truth
                ids,sizes = model.dump_gt(targets,boxfield)
                
                targets = [{k: v.to(device) for k, v in t.items() if k != 'id'} for t in targets]
                box_out,cap_out = model(detections, captions)
                model.dump_dt(box_out,ids,sizes)
                captions_gt = captions[:, 1:].contiguous()
                cap_out = cap_out[:, :-1].contiguous()
                loss_cap = loss_fn(cap_out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                this_loss = loss_cap.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
    mAP = 0
    if e % 4 == 0:
        mAP = model.evalue_box_(args)
    val_loss = running_loss / len(dataloader)
    return val_loss,mAP


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, targets, captions) in enumerate(iter(dataloader)):#(images, caps_gt)
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(captions, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    print('lr = ', optim.state_dict()['param_groups'][0]['lr'])
    
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:#x1,y1,x2,y12
        for it, (detections, targets, captions) in enumerate(dataloader):
            targets = [{k: v.to(device) for k, v in t.items() if k != 'id'} for t in targets]
            detections,  captions = detections.to(device), captions.to(device) #batch len dim ,batch*

            box_out,cap_out = model(detections, captions)#25,100,4 /// 25,100,1602

            captions_gt = captions[:, 1:].contiguous()
            cap_out = cap_out[:, :-1].contiguous()
            loss_cap = loss_fn(cap_out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            loss_box = model.forward_box_loss(box_out,targets)
            
            loss = args.norm_r * loss_box + (1-args.norm_r) * loss_cap
            # loss = loss_cap
            optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optim.step()

            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0

    model.train()
    scheduler_rl.step()
    print('lr = ', optim_rl.state_dict()['param_groups'][0]['lr'])

    running_loss = .0
    seq_len = 20
    
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections,targets, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'id'} for t in targets] if args.box_in_lr else targets
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            if args.box_in_lr:
                box_out = model(detections, caps_gt, only_box = True)

                
            
            optim.zero_grad()

            # Rewards
            if args.box_in_lr:
                loss_box = model.forward_box_loss(box_out,targets)

            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()

            if args.box_in_lr:
                loss = loss + loss_box * 0.1 * args.norm_r
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))

            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Experiment_Train')
    parser.add_argument('--exp_name', type=str, default='Experiment')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--position_embedding', type=str, default='sine',help='sine or learned')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')

    parser.add_argument('--features_path', type=str, default='Datasets/X101-features/coco_X101_grid.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='Datasets/m2_annotations/')
    parser.add_argument('--dir_to_save_model', type=str, default='saved_transformer_models/')
    parser.add_argument('--logs_folder', type=str, default='transformer_tensorboard_logs')
    parser.add_argument('--path_txtlog',type=str,default='log')
    parser.add_argument('--dect_path', type=str, default='')
    parser.add_argument('--image_path', type=str, default='Datasets')

    parser.add_argument('--path_prefix',type=str,default='/home/awen/workstation/dataset/rstnet')
    #/media/a100202/ccc739a0-163b-4b54-b335-f12f0d52de59/zhangawen/dataset/rstnet
    parser.add_argument('--path_prefix_web',type=str,default='/media/a1002/8b95f0e0-6f6d-4dcb-a09a-a0272b8be2b7/zhangawen/rstnet')
    parser.add_argument('--path_vocab',type=str,default='vocab.pkl')
    
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=30)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)
    parser.add_argument('--xe_base_lr', type=float, default=0.0001)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--norm_r', type=float, default=0.5)
    #evalue_box
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", default=True)
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", default=True)
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")

    #参数调整
    parser.add_argument('--id', type=str, default='default')
    parser.add_argument('--model', type=int, default=5)
    parser.add_argument('--web', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--aux_outputs', type=bool, default=False)
    parser.add_argument('--box_in_lr', type=bool, default=False)
    parser.add_argument('--add_features', type=bool, default=False)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    
    print("现在正在使用的GPU编号:", end="")
    print(torch.cuda.current_device())
    
#*******************************************************************************
    if args.web:
        args.path_prefix = args.path_prefix_web

    path_ = ['image_path', 'features_path', 'annotation_folder', 'dir_to_save_model', 'logs_folder', 'path_vocab', 'path_txtlog', 'dect_path']

    for p in path_ :
        setattr(args, p, os.path.join(args.path_prefix, getattr(args, p)))

#*******************************************************************************
    print(args)
    print('The Training of Experiment')

    #init textlog
    log = Logger(args.id+"_" + str(datetime.today().date()), args.path_txtlog)
    log.write_log("args setting:\n"+str(args)+"\n")
    log.write_log("****************************init******************************\n")

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # init vocab
    print('Loading from vocabulary')
    text_field.vocab = pickle.load(open(args.path_vocab, 'rb'))

    #build dataset,dataloader
    datasets, datasets_evalue,boxfield = Build_DataSet(args=args, text_field=text_field)

    #build model
    model = BuildModel.build(args.model, args).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n model size: %d\n' % n_parameters)

    ref_caps_train = list(datasets['train'].text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    '''
    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    '''

    def lambda_lr(s):
        print("s:", s)
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        refine_epoch = args.refine_epoch_rl 
        print("rl_s:", s)
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr


    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])

    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name)
        else:
            fname = os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if use_rl:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            else:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):

        dataloader_train = DataLoader(dataset=datasets['train'], collate_fn=datasets['train'].collate_fn(),
                                      batch_size=args.batch_size, shuffle=True,num_workers=args.workers,pin_memory=True)
        dataloader_val = DataLoader(dataset=datasets['val'], collate_fn=datasets['val'].collate_fn(),
                                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dataset=datasets_evalue['e_train'], collate_fn=datasets_evalue['e_train'].collate_fn(),
                                           batch_size=args.batch_size//5, shuffle=True, num_workers=args.workers,pin_memory=True)
        dict_dataloader_val = DataLoader(dataset=datasets_evalue['e_val'], collate_fn=datasets_evalue['e_val'].collate_fn(),
                                         batch_size=args.batch_size//5, shuffle=False, num_workers=args.workers)
        dict_dataloader_test = DataLoader(dataset=datasets_evalue['e_test'], collate_fn=datasets_evalue['e_test'].collate_fn(),
                                          batch_size=args.batch_size//5, shuffle=False, num_workers=args.workers)

        log.write_log('epoch%d:\n' % e)

        if  use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)

            log.write_log('state = %s \n' % 'base_train')
            log.write_log(' train_loss = %f \n' % train_loss)
            
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field)


            log.write_log('state = %s \n' % 'rl_train')
            log.write_log(' train_loss = %f \n'%train_loss)
            log.write_log(' reword = %f \n' % reward)
            log.write_log(' reward_baseline = %f \n' % reward_baseline)

        
        # Validation loss
        val_loss ,mAP= evaluate_loss(model, dataloader_val, loss_fn, text_field)


        print(' mAP = %f \n' % mAP)
        log.write_log(' mAP = %f \n' % mAP)
        log.write_log(' val_loss = %f \n' % val_loss)
        log.write_log("\n")

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']


        log.write_log('val_evaluation scores = %s' % str(scores))
        log.write_log("\n")

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']


        log.write_log('test_evaluation scores = %s' % str(scores))
        log.write_log("\n")

        log.write_log("************************epoch %d end**************************\n" % e)
        log.write_log("**************************************************************\n")
        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if e < args.xe_least:   # xe stage train 15 epoches at least 
                print('special treatment, e = {}'.format(e))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                
                for k in range(e-1):
                    scheduler_rl.step()

                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if e == args.xe_most:     # xe stage no more than 20 epoches
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

                for k in range(e-1):
                    scheduler_rl.step()

                print("Switching to RL")

        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, best_cider %f, and best test_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider'], data['best_test_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name))
        
        if best:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
        if best_test:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best_test.pth' % args.exp_name))


#         # 保存模型，用于微调
#         if e >= 25:
#             copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '{}_{}.pth'.format(args.exp_name, e)))

        if exit_train:
            # writer.close()
            break