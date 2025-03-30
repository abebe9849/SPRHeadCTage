import hydra
from omegaconf import DictConfig, OmegaConf
import sys,gc,os,random,time,math,glob
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
from  torch.cuda.amp import autocast, GradScaler 
import cv2,timm
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.metrics import log_loss
from functools import partial
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,f1_score,log_loss,mean_absolute_error,mean_squared_error
from  sklearn.metrics import accuracy_score as acc
import torch
import torch.nn as nn
from torch.optim import Adam, SGD,AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,RandomGamma, RandomRotate90,GaussNoise,RandomBrightnessContrast,Resize
from albumentations.pytorch import ToTensorV2
import transformers as T

import albumentations as A
#import vision_transformer as vits

### my utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d


# From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p)


def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


_GEM_FN = {
    1: gem_1d, 2: gem_2d, 3: gem_3d
}


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return _GEM_FN[self.dim](x, p=self.p, eps=self.eps)


class AdaptiveConcatPool1d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool1d(x, 1), F.adaptive_max_pool1d(x, 1)), dim=1)


class AdaptiveConcatPool2d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)), dim=1)
"""
age=age/100
とする　０〜１となったものをbceで

"""
###

import logging
#from mylib.
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


from timm.data.transforms import RandomResizedCropAndInterpolation
def select_random_elements(list_a, N):
    if len(list_a) < N:
        # 重複を許して要素を選択
        return random.choices(list_a, k=N)
    else:
        # 重複なしで要素を選択
        return random.sample(list_a, N)
def pad_to_square_bbox(x, y, w, h, image_width, image_height):
    """
    与えられたバウンディングボックス(xywh形式)を、長辺に対してパディングして正方形にします。
    :param x: バウンディングボックスの左上のX座標
    :param y: バウンディングボックスの左上のY座標
    :param w: バウンディングボックスの幅
    :param h: バウンディングボックスの高さ
    :param image_width: 画像の幅
    :param image_height: 画像の高さ
    :return: パディング後のバウンディングボックス（xywh形式）
    """
    # 長辺を求める
    max_side = max(w, h)

    # 新しい幅と高さを長辺に設定
    new_w = new_h = max_side

    # 新しいX、Y座標を計算
    new_x = x - (new_w - w) / 2
    new_y = y - (new_h - h) / 2

    # バウンディングボックスが画像の範囲を超えないように調整
    new_x = max(0, min(new_x, image_width - new_w))
    new_y = max(0, min(new_y, image_height - new_h))

    return int(new_x), int(new_y), int(new_w), int(new_h)

# 例: 使用例
x, y, w, h = 50, 50, 100, 150  # 元のバウンディングボックス
image_width, image_height = 500, 500  # 画像のサイズ
new_x, new_y, new_w, new_h = pad_to_square_bbox(x, y, w, h, image_width, image_height)
print(new_x, new_y, new_w, new_h)

      
def crop_func(img,xywh_):
    x,y,w,h,_ = map(int,xywh_)
    max_side = max(w, h)

    # 新しい幅と高さを長辺に設定
    new_w = new_h = max_side
    image_width,image_height,_ = img.shape

    # 新しいX、Y座標を計算
    new_x = x - (new_w - w) / 2
    new_y = y - (new_h - h) / 2
    new_x = int(max(0, min(new_x, image_width - new_w)))
    new_y = int(max(0, min(new_y, image_height - new_h)))
    
    #return img[y:y+h,x:x+w]
    return img[new_y:new_y+new_h,new_x:new_x+new_w]


class TrainDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG
        self.train = train
        self.image_size_seg = (128, 128, CFG.N_patch)
        self.ids = self.df["StudyID___SeriesID"].values
        self.labels = self.df["Age"].values
        self.root = "../data"
        paths_ = [glob.glob(f"{self.root}/{ct_id}/*.png") for ct_id in self.ids]
        self.id_2_paths = dict(zip(self.ids, paths_))
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ct_id = self.ids[idx]
        img_paths = self.id_2_paths[ct_id]
        crop_ = [np.load(p.replace("25d","25d_xywh").replace(".png",".npy")) for p in img_paths]

        img_paths = [[i,j] for i,j in zip(img_paths,crop_) if j[-1]>0.05]
        crop_ = np.array(crop_)
        max_xywh = crop_[np.argmax(crop_[:,4])]
        #print(img_paths)
        indices = np.quantile(list(range(len(img_paths))), np.linspace(0., 1., self.image_size_seg[2])).round().astype(int)
        img_paths = [img_paths[i] for i in indices]
        if self.CFG.precrop:
            imgs= [crop_func(cv2.imread(i[0]),max_xywh) for i in img_paths]
        else:
            imgs = [cv2.imread(i[0]) for i in img_paths]
        imgs =  np.stack([self.transform(image=img)['image']  for img in imgs])
        image = torch.from_numpy(imgs.transpose(0,3,1,2)).float()

        label =  self.labels[idx]
        if self.CFG.loss.name=="BCE":
            label = label/100.
            label = torch.tensor(label).float()
        elif self.CFG.loss.name=="MSE":
            label = torch.tensor(label).float()
        elif self.CFG.loss.name in ["CE","DLDL"]:
            label = torch.tensor(label).long()

        
        return image, label




#### dataset ==============

#### augmentation ==============


def get_transforms(*, data,CFG):
    if data == 'train':
        return Compose([
            Resize(CFG.preprocess.size,CFG.preprocess.size),
            #A.augmentations.crops.transforms.CenterCrop(CFG.preprocess.size*0.9,CFG.preprocess.size*0.9),
            #A.crops.transforms.RandomResizedCrop(CFG.preprocess.size,CFG.preprocess.size,scale=(0.5, 1.0)),
            #A.crops.transforms.RandomCrop(CFG.preprocess.size,CFG.preprocess.size),
            A.HorizontalFlip(p=CFG.aug.HorizontalFlip.p),
            A.VerticalFlip(p=CFG.aug.VerticalFlip.p),
            A.RandomRotate90(p=CFG.aug.RandomRotate90.p),
            A.ShiftScaleRotate(
                shift_limit=CFG.aug.ShiftScaleRotate.shift_limit,
                scale_limit=CFG.aug.ShiftScaleRotate.scale_limit,
                rotate_limit=CFG.aug.ShiftScaleRotate.rotate_limit,
                p=CFG.aug.ShiftScaleRotate.p),
            A.RandomBrightnessContrast(
                brightness_limit=CFG.aug.RandomBrightnessContrast.brightness_limit,
                contrast_limit=CFG.aug.RandomBrightnessContrast.contrast_limit,
                p=CFG.aug.RandomBrightnessContrast.p),
            A.CLAHE(
                clip_limit=(1,4),
                p=CFG.aug.CLAHE.p),
            A.OneOf([
                A.ImageCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=CFG.aug.compress.p),
            #A.CoarseDropout(max_holes=CFG.aug.CoarseDropout.max_holes, max_height=CFG.aug.CoarseDropout.max_height, max_width=CFG.aug.CoarseDropout.max_width, p=CFG.aug.CoarseDropout.p),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'valid':
        return Compose([
            Resize(CFG.preprocess.size,CFG.preprocess.size),
            #A.augmentations.crops.transforms.CenterCrop(int(CFG.preprocess.size*0.9),int(CFG.preprocess.size*0.9)),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

def dldl_v2_loss(logits: torch.tensor, labels: torch.tensor, sigma: float = 2, lambda_: float = 1):
    """
    Computes the loss as defined in Learning Expectation of Label Distribution for Facial Age and Attractiveness Estimation.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        sigma (float, optional): Standard deviation of the Gaussian GT label distribution. Defaults to 2.
        lambda_ (float, optional): Weight of L1 loss. Defaults to 1.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    probas = nn.functional.softmax(logits, dim=1)
    diffs = torch.diff(probas, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)
    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    sigmas = torch.ones_like(labels).to(labels.device)*sigma
    broadcast_sigmas = torch.broadcast_to(
        sigmas[:, None], probas.shape).to(labels.device)
    label_distributions = torch.exp(-((class_labels - broadcast_labels)**2)/(
        2*broadcast_sigmas**2)) / (torch.sqrt(2*torch.pi*broadcast_sigmas))
    label_distributions = label_distributions / torch.broadcast_to(
        torch.sum(label_distributions, dim=1, keepdim=True), label_distributions.shape)

    means = torch.sum(probas*class_labels, dim=1)

    loss = nn.functional.cross_entropy(logits, label_distributions) + \
        lambda_ * torch.mean(torch.abs(means-labels))
    return loss

class DLDL2_loss(nn.Module):
    
    def __init__(self, sigma=2, lambda_=1):
        super().__init__()
        self.sigma = sigma
        self.lambda_ = lambda_

    def forward(self, x, target):
        loss = dldl_v2_loss(x,target,self.sigma,self.lambda_)
        return loss
#### augmentation ==============
from transformers import AutoProcessor, CLIPVisionModel
#### model ================
SEQ_POOLING = {
    'gem': GeM(dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': nn.AdaptiveAvgPool2d(1),
    'max': nn.AdaptiveMaxPool2d(1)
}
dic_NUM_CLS = {"BCE":1,"CE":100,"MSE":1,"DLDL2":100}

class Model_iafoss(nn.Module):
    def __init__(self,CFG, base_model='tf_efficientnet_b0_ns',pool="avg",pretrain=True):
        super(Model_iafoss, self).__init__()
        self.base_model = base_model 
        NUM_CLS = dic_NUM_CLS[CFG.loss.name]
        """
        if self.base_model in ["hipt","plip","qnet","ibot"]:
            if self.base_model=="ibot":
                checkpoint_key = "teacher"
                pretrained_weights = "/home/abe/pandasub/ibot/pandaExp000/checkpoint.pth"
                self.model = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
                state_dict = torch.load(pretrained_weights, map_location="cpu")
                if checkpoint_key is not None and checkpoint_key in state_dict:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                    state_dict = state_dict[checkpoint_key]
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict, strict=False)
                for _, p in self.model.named_parameters():
                    p.requires_grad = False
                for _, p in self.model.head.named_parameters():
                    p.requires_grad = True

                freeze =9
                for n, p in self.model.blocks.named_parameters():
                    if int(n.split(".")[0])>=(12-freeze):
                        p.requires_grad = True
                        
                self.n_last_blocks  = 4
                avgpool_patchtokens = 0
                
                nc = self.model.embed_dim * (self.n_last_blocks + int(avgpool_patchtokens))
            
            
            if self.base_model=="plip":
                self.model = CLIPVisionModel.from_pretrained("vinid/plip")
                nc = 768
            self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
            nc*=CFG.N_patch
            self.head = nn.Sequential(nn.Linear(nc,512),
                            nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,3))
            self.exam_predictor = nn.Linear(512*2, 3)
            self.pool = nn.AdaptiveAvgPool1d(1)
        """
        if self.base_model=="dino":
            print("not implemet")
            exit()
        else:
            self.model = timm.create_model(self.base_model, pretrained=True, num_classes=0,in_chans=3)
            """
            #not work grad_accm not work
            for module in self.model.modules():
                
                if isinstance(module, timm.models.layers.BatchNormAct2d):
                    #print(module)
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
            """

            #self.model.conv_stem = nn.Conv2d(2, 32, kernel_size=3, padding=1, stride=1, bias=False)
            nc = self.model.num_features
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(nc,512),
                            nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,NUM_CLS))

            self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
            self.exam_predictor = nn.Linear(512*2, 1)
            self.pool = nn.AdaptiveAvgPool1d(1)

        
    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]

        input1 = input1.view(-1,shape[2],shape[3],shape[4])

        if "ibot" in self.base_model:
            intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
            x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
            #x = x.view(batch_size,x.shape[1]*n)
            #y = self.head(x)
           
           #python base.py model.name="ibot" train.lr=0.0001
            return y
        elif self.base_model=="plip":
            x = self.model(input1)["pooler_output"]
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
           
            return y
        elif self.base_model=="qnet":
            x = self.model.encode_image(input1)
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
           
            return y
            
        else:
            #"""
            x = self.model.forward_features(input1)#bs*num_tile,embed_dim,h,w
            shape = x.size()
            x = x.view(-1,n,shape[1],shape[2],shape[3])
            x = x.permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
            y = self.head(x)
            """
            x =  self.model(input1)
            
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
            """
            return y

#model = Model_iafoss("dino_vit_s")



#### model ================


def train_fn(CFG,fold,folds,test_pl=0):


    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"### fold: {fold} ###")
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    
    val_folds = folds.loc[val_idx]
    tra_folds = folds.loc[trn_idx]
    if CFG.general.debug:
        CFG.train.epochs =2
        tra_folds = tra_folds[tra_folds["StudyID___SeriesID"].isin(tra_folds["StudyID___SeriesID"].unique()[:50])]
        val_folds = val_folds[val_folds["StudyID___SeriesID"].isin(val_folds["StudyID___SeriesID"].unique()[:50])]

        
    print(val_folds["StudyID___SeriesID"].nunique(),tra_folds["StudyID___SeriesID"].nunique())
    if type(test_pl)!=type(0):
        tra_folds = pd.concat([tra_folds,test_pl]).reset_index(drop=True)

    train_dataset = TrainDataset(tra_folds.reset_index(drop=True),train=True, transform1=get_transforms(data='train',CFG=CFG),CFG=CFG)#get_transforms(data='train',CFG=CFG)
    valid_dataset = TrainDataset(val_folds.reset_index(drop=True),train=False,transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)#


    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=8,pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=8,pin_memory=True)

    ###  model select ============
    model = Model_iafoss(CFG,base_model=CFG.model.name).to(device)
    # ============


    ###  optim select ============
    if CFG.train.optim=="adam":
        optimizer = Adam(model.parameters(), lr=CFG.train.lr, amsgrad=False)
    elif CFG.train.optim=="adamw":
        optimizer = AdamW(model.parameters(), lr=CFG.train.lr,weight_decay=5e-5)
    # ============

    ###  scheduler select ============
    if CFG.train.scheduler.name=="cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs, eta_min=CFG.train.scheduler.min_lr)
    elif CFG.train.scheduler.name=="cosine_warmup":
        scheduler =T.get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=len(train_loader)*CFG.train.scheduler.warmup,
        num_training_steps=len(train_loader)*CFG.train.epochs)

    # ============

    ###  loss select ============
    if CFG.loss.name=="BCE":
        criterion=nn.BCEWithLogitsLoss()
    elif CFG.loss.name=="CE":
        criterion=nn.CrossEntropyLoss()
    elif CFG.loss.name=="DLDL2":
        criterion=DLDL2_loss()
    elif CFG.loss.name=="MSE":
        criterion=nn.HuberLoss() #https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/277690
        
    print(criterion)
    ###  loss select ============

    scaler = torch.cuda.amp.GradScaler()
    best_score = np.inf
    best_loss = np.inf
    best_preds = None
        
    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            

            ### mix系のaugumentation=========

            ### mix系のaugumentation おわり=========

            if CFG.train.amp:
                with autocast():
                    y_preds = model(images)
                    if CFG.loss.name=="BCE" or CFG.loss.name=="MSE":
                        loss_ = criterion(y_preds,labels.view(-1,1))
                    else:
                        loss_ = criterion(y_preds,labels)
                    loss=loss_

                scaler.scale(loss).backward()

                if (i+1)%CFG.train.ga_accum==0 or i==-1:
                    scaler.step(optimizer)
                    scaler.update()
                    if CFG.train.scheduler.name=="cosine_warmup":
                        scheduler.step()
            if CFG.train.scheduler.name=="cosine":
                scheduler.step()
            avg_loss += loss.item() / len(train_loader)
        model.eval()
        avg_val_loss = 0.
        LOGITS = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))
        RANK = torch.Tensor([i for i in range(100)]).to(device)
        for i, (images, labels) in tk1:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                with autocast(enabled=False):
                    logits = model(images)
                    if CFG.loss.name=="BCE" or CFG.loss.name=="MSE":
                        loss_ = criterion(logits,labels.view(-1,1))
                    else:
                        loss_ = criterion(logits,labels)
            valid_labels.append(labels)
            if CFG.loss.name=="BCE":
                LOGITS.append(logits.detach().sigmoid())
            elif CFG.loss.name=="MSE":
                LOGITS.append(logits.detach())
            elif CFG.loss.name in ["CE","DLDL2"]:
                logits = nn.functional.softmax(logits.detach(), dim=1)
                
                LOGITS.append(torch.sum(logits * RANK, dim=1))
                
            avg_val_loss += loss.item() / len(valid_loader)
        
        preds = torch.cat(LOGITS).cpu().numpy().squeeze()
        valid_labels = torch.cat(valid_labels).cpu().numpy()
        if CFG.loss.name=="BCE":
            preds*=100
            valid_labels*=100


        #each_auc,score =AUC(true=valid_labels,predict=preds)
        MAE_score = mean_absolute_error(valid_labels, preds)


        elapsed = time.time() - start_time
        log.info(f"MAE  {MAE_score}")


        log.info(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s')

        #if best_loss>avg_val_loss:#pr_auc best
        #    best_loss = avg_val_loss
        #    log.info(f'  Epoch {epoch+1} - Save Best loss: {best_loss:.4f}')
        #    torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_best_loss.pth')

        if best_score>MAE_score:#pr_auc best
            best_score = MAE_score
            log.info(f'  Epoch {epoch+1} - Save Best MAE: {best_score:.4f}')
            best_preds = preds
            torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_best_MAE.pth')


    return best_preds, valid_labels



def eval_func(model, valid_loader, device,CFG):
    model.to(device) 
    model.eval()

    valid_labels = []
    preds = []

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, (images, labels) in tk1:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            with autocast():
                y_preds = model(images.float())
                y_preds = y_preds.sigmoid()

        valid_labels.append(labels.to('cpu').numpy())
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    return preds,valid_labels



def inf_func(models, valid_loader, device,CFG):
    for model in models:
        model.eval()

    preds = []
    RANK = torch.Tensor([i for i in range(100)]).to(device)
    

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, (images, _) in tk1:
        images = images.to(device,non_blocking=True)
        with torch.no_grad():
            with autocast():
                if CFG.loss.name=="BCE":
                    y_preds = [m(images.float()).detach().sigmoid() for m  in models]
                elif CFG.loss.name=="MSE":
                    y_preds = [m(images.float()).detach() for m  in models]
                elif CFG.loss.name in ["CE","DLDL2"]:
                    y_preds = [torch.sum(nn.functional.softmax(m(images.float()).detach(), dim=1) * RANK, dim=1) for m  in models]                    
                    
                y_preds  = torch.stack(y_preds,axis=-1)#.median(0)
                #https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/276138

        preds.append(y_preds)
        
    preds = torch.cat(preds).cpu().numpy().squeeze()
    if CFG.loss.name=="BCE":
        preds*=100

    return preds
        

    
def submit(CFG,num_folds,test,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for fold in   range(num_folds):
        model = Model_iafoss(CFG,base_model=CFG.model.name).to(device)
        model.load_state_dict(torch.load(f"{DIR}/fold{fold}_{CFG.general.exp_num}_best_MAE.pth", map_location="cpu"))
        models.append(model)
        
    valid_dataset = TrainDataset(test,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=12,pin_memory=True)
    tets_preds = inf_func(models, valid_loader, device,CFG)
    print(tets_preds.shape)
    for i in range(num_folds):
        col = f"pred_{i}"
        test[col]=tets_preds[:,i]
    
    
    return test


def calculate_median(df):
# predカラムの値を1つのリストに集約
    preds = df.filter(like='pred').values.flatten()
    # 中央値を計算
    return np.median(preds)

DIR = "."

log = logging.getLogger(__name__)
@hydra.main(config_path=f"{DIR}/",config_name="base")
def main(CFG : DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)

    log.info(f"===============exp_num{CFG.general.exp_num}============")

    folds = pd.read_csv("../data/25d_10folds.csv")
    num_folds = int(folds["fold"].max())+1

    #"""
    preds = []
    valid_labels = []
    oof = pd.DataFrame()
    
    if CFG.psuedo_label!=0:
        test_pl=pd.read_csv(CFG.psuedo_label)
        test_pl["fold"]=999
    else:
        test_pl = 0
    #time.sleep(3600*4)

        
    
    
    for fold in range(num_folds):
        _preds, _valid_labels = train_fn(CFG,fold,folds,test_pl)
        preds.append(_preds)
        valid_labels.append(_valid_labels)
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    MAE_score = mean_absolute_error(valid_labels, preds)
    log.info(f"OOF MAE_score  {MAE_score}")


    #oof.to_csv(f"oof_{CFG.general.exp_num}.csv",index=False)
    test_df = pd.read_csv("../data/25d_test.csv")
    test_df["Age"]=0
    test_df = submit(CFG,num_folds,test_df,DIR=".")

    test_df.to_csv(f"inf_{CFG.general.exp_num}.csv",index=False)
    test_df = test_df.groupby('StudyID').apply(calculate_median).reset_index(name='Age')
    test_df["StudyID"] = [str(i).zfill(6) for i in test_df["StudyID"].values]
    test_df["Age"] =test_df["Age"].round()

    
    test_df.to_csv(f"test_{CFG.general.exp_num}.csv",index=False)


if __name__ == "__main__":
    main()
