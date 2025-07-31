# ======================== #
#         Imports          #
# ======================== #
import os
import math
import time
import random
import argparse
import configparser
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.stid import STID
from models.vis import CVModel
from lib.utils import log_string, loadData, _compute_loss, metric, ImageFolderDataset
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ======================== #
#          Solver          #
# ======================== #
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        train_dataset, val_dataset, test_dataset,\
        self.mean, self.std,   self.tmean, self.tstd, \
        self.position_relat, self.locations = loadData(
            self.traffic_file, self.meta_file, 
            self.input_len, self.output_len,
            self.train_ratio, self.test_ratio,
            self.adj_file,
            self.tod, self.dow,
            log
        )
        dataset = ImageFolderDataset(self.image_file,self.img_size)
        self.img_loader = DataLoader(dataset, batch_size=self.img_batch_size, num_workers=8, pin_memory=False)
        
        self.position_relat = torch.tensor(self.position_relat)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=8, pin_memory=True)


        log_string(log, '------------ End -------------\n')

        self.best_epoch = 0
        self.writer = SummaryWriter(log_dir=self.log_file)
        self.build_model()

    def build_model(self):
        LC = torch.from_numpy(self.locations).to(self.device).float()
        LC = (LC - torch.min(LC)) / (torch.max(LC) - torch.min(LC) + 1e-8)

        self.cvmodel = CVModel(self.device, self.position_relat, hidden_dim=self.hidden_dim, img_h=self.img_size, img_w=self.img_size,cv_fea_dim = self.cv_fea_dim, support_num=self.support_num, backbone=self.backbone).to(self.device)
        
        self.model  = STID(self.device, num_nodes=self.node_num,layers=self.layers,support_num =self.support_num,
                           hidden_dim=self.hidden_dim,node_dim=self.node_dim, tod_dims=self.tod_dims,dow_dims=self.dow_dims, cvrelation_dim = self.cv_fea_dim, lc=LC,dropout=self.dropout, partten_dim = self.partten_dim, act = self.active,
                           cv_token=self.cv_token, cv_pattern=self.cv_pattern, relation_patterns = self.relation_patterns, st_encoder = self.st_encoder, node_cv_num = self.node_cv_num).to(self.device)

        self.optimizerst = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.optimizercv = torch.optim.AdamW(self.cvmodel.parameters(),
                                        lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.lr_schedulerst = torch.optim.lr_scheduler.StepLR(
            self.optimizerst,
            step_size=self.step_size,  
            gamma=0.5      
        )
        self.lr_schedulercv = torch.optim.lr_scheduler.StepLR(
            self.optimizercv,
            step_size=self.step_size,  
            gamma=0.5    
        )
        

    def vali(self):
        self.model.eval()
        self.cvmodel.eval()

        pred, label = [], []
        supports, static_fea = [], []
        vision_rela = []
        
        with torch.no_grad():
            for img_idx, img_batch in enumerate(self.img_loader):
                start_idx = img_idx * self.img_batch_size
                end_idx = min(self.node_num, start_idx + self.img_batch_size)
                img = img_batch.to(self.device)
                out, sfea, sp, kk  = self.cvmodel(img, start_idx, end_idx, self.model.node_emb)  # b h w c
                static_fea.append(sfea.squeeze())
                supports.append(sp)
                vision_rela.append(kk)
            
            static_fea = torch.cat(static_fea, 0)  # [N, D]
            vision_rela = torch.cat(vision_rela, dim=0)  # [NC, D]
            
            if self.support_num:
                supports = list(torch.cat(supports, 0).permute(2, 0, 1))
            else:
                supports = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                X, Y, TE, _ = [b.float().to(self.device) for b in batch]
                Y = Y.cpu().numpy()
                
                NormX = (X - self.mean) / self.std
                NormX = NormX.transpose(1, 3)
                TE = TE.transpose(1, 3)
                
                if not self.cv_token:
                    y_hat = self.model(NormX, TE, supports, None, vision_rela)
                else:
                    y_hat = self.model(NormX, TE, supports, static_fea, vision_rela)
                
                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(Y)
                
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        maes, rmses, mapes = [], [], []

        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, f'step {i + 1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')

        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        
        log_string(log, f'average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def get_next_img_batch(self):
        if not hasattr(self, 'img_iter'):
            self.img_iter = enumerate(iter(self.img_loader))

        try:
            batch_idx, img = next(self.img_iter)
        except StopIteration:
            self.img_iter = enumerate(iter(self.img_loader))
            batch_idx, img = next(self.img_iter)

        img = img.to(self.device)
        cv_start_index = batch_idx * self.img_batch_size
        cv_end_index = min(self.node_num, cv_start_index + self.img_batch_size)

        return img, cv_start_index, cv_end_index

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = float('inf')

        for epoch in range(1, self.max_epoch + 1):
            train_l_sum, batch_count, start = 0.0, 0, time.time()
            num_batch = len(self.train_loader)
            
            self.model.train()
            self.cvmodel.train()

            num_vis_node = self.cvmodel.num_vis_node
            supports, static_fea, vision_rela = [], [], []
            with torch.no_grad():
                for img_idx, img_batch in enumerate(self.img_loader):
                    start_idx = img_idx * self.img_batch_size
                    end_idx = min(self.node_num, start_idx + self.img_batch_size)
                    img = img_batch.to(self.device)
                    out, sfea, sp, kk  = self.cvmodel(img, start_idx, end_idx, self.model.node_emb)  # b h w c
                    static_fea.append(sfea.squeeze())
                    supports.append(sp)
                    vision_rela.append(kk)
                static_fea = torch.cat(static_fea, 0) #[N, D]
                vision_rela = torch.cat(vision_rela, dim=0)  # [NC, D]
                if self.support_num:
                    supports = list(torch.cat(supports, 0).permute(2, 0, 1))
                else:
                    supports = []

            with tqdm(total=num_batch) as pbar:
                for batch_idx, batch in enumerate(self.train_loader):
                    X, Y, TE, _ = [b.float().to(self.device) for b in batch]
                    
                    if self.support_num:
                        supports = [list(x.detach().requires_grad_(True)) for x in supports]
                    else:
                        supports = []
                    static_fea = list(static_fea.detach().requires_grad_(True))
                    vision_rela = list(vision_rela.detach().requires_grad_(True))
                    img, cv_start_index, cv_end_index = self.get_next_img_batch()
                    out, center_fearure, sp, kk  = self.cvmodel(img, cv_start_index, cv_end_index, self.model.node_emb)  # b h w c
                    
                    static_fea[cv_start_index : cv_end_index] = center_fearure.squeeze()
                    static_fea = torch.stack(static_fea, 0)
                    vision_rela[num_vis_node[cv_start_index]:num_vis_node[cv_start_index] + len(kk)] = kk
                    vision_rela = torch.stack(vision_rela, dim=0)
                    if self.support_num:
                        sp = sp.permute(2, 0, 1) 
                        for i in range(self.support_num):
                            for k, j in enumerate(range(cv_start_index, cv_end_index)):
                                supports[i][j] = sp[i,k] 
                        supports = [torch.stack(x,dim=0) for x in supports]
                    else:
                        supports = []

                    NormX = (X - self.mean) / self.std
                    
                    NormX = NormX.transpose(1, 3)
                    TE = TE.transpose(1, 3)

                    self.optimizerst.zero_grad()
                    self.optimizercv.zero_grad()
                    
                    if not self.cv_token:
                        y_hat = self.model(NormX, TE, supports, None, vision_rela)
                    else:
                        y_hat = self.model(NormX, TE, supports, static_fea, vision_rela)

                    loss = _compute_loss(Y, y_hat* self.std + self.mean)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizerst.step()
                    self.optimizercv.step()
                    train_l_sum += loss.item()
                    batch_count += 1
                    pbar.update(1)

            self.writer.add_scalar('1_Training/loss', train_l_sum / batch_count, epoch)
            self.writer.add_scalar('1_Training/lr', self.optimizerst.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('1_Training/time', time.time() - start, epoch)

            log_string(log, f'epoch {epoch}, lr {self.optimizerst.param_groups[0]["lr"]:.6f}, '
                             f'loss {train_l_sum / batch_count:.4f}, time {time.time() - start:.1f} sec')
            self.lr_schedulerst.step()
            self.lr_schedulercv.step()
            

            if epoch % 2 == 0:
                mae, rmse, mape = self.vali()
                self.writer.add_scalar('2_Validation/MAE', mae[-1], epoch)
                self.writer.add_scalar('2_Validation/RMSE', rmse[-1], epoch)
                self.writer.add_scalar('2_Validation/MAPE', mape[-1], epoch)

                if mae[-1] < min_loss:
                    self.best_epoch = epoch
                    min_loss = mae[-1]
                    torch.save(self.model.state_dict(), self.model_file)
                    torch.save(self.cvmodel.state_dict(), self.cvmodel_file)

        self.writer.close()
        log_string(log, f'Best MAE is: {min_loss:.4f}')
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.cvmodel.load_state_dict(torch.load(self.cvmodel_file, map_location=self.device))
        self.model.eval()
        self.cvmodel.eval()
        
        pred, label = [], []
        supports, static_fea, vision_rela = [], [], []
        
        with torch.no_grad():
            for img_idx, img_batch in enumerate(self.img_loader):
                start_idx = img_idx * self.img_batch_size
                end_idx = min(self.node_num, start_idx + self.img_batch_size)
                img = img_batch.to(self.device)
                out, sfea, sp, kk  = self.cvmodel(img, start_idx, end_idx, self.model.node_emb)  # b h w c

                static_fea.append(sfea.squeeze())
                supports.append(sp)
                vision_rela.append(kk)
            static_fea = torch.cat(static_fea, 0) #[N, D]
            vision_rela = torch.cat(vision_rela, dim=0)  # [NC, D]
            if self.support_num:
                supports = list(torch.cat(supports, 0).permute(2, 0, 1))
            else:
                supports = []
    

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                X, Y, TE, _ = [b.float().to(self.device) for b in batch]
                Y = Y.cpu().numpy()

                NormX = (X - self.mean) / self.std
                NormX = NormX.transpose(1, 3)
                TE = TE.transpose(1, 3)
                if not self.cv_token:
                    y_hat = self.model(NormX, TE, supports, None, vision_rela)
                else:
                    y_hat = self.model(NormX, TE, supports, static_fea, vision_rela)

                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(Y)

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)

        maes, rmses, mapes = [], [], []

        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, f'step {i + 1}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')

        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)

        log_string(log, f'average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')

        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", type=str, required=True, help='Path to configuration file')
    parser.add_argument('--backbone', type=str, default="resnet", help='Model backbone')

    args, unknown = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(args.config)


    parser.add_argument('--cuda', type=str, default=config.get('train', 'cuda'))
    parser.add_argument('--seed', type=int, default=config.getint('train', 'seed'))
    parser.add_argument('--batch_size', type=int, default=config.getint('train', 'batch_size'))
    parser.add_argument('--img_size', type=int, default=config.getint('train', 'img_size'))
    parser.add_argument('--img_batch_size', type=int, default=config.getint('train', 'img_batch_size'))
    parser.add_argument('--switch_interval', type=int, default=config.getint('train', 'switch_interval'))
    parser.add_argument('--max_epoch', type=int, default=config.getint('train', 'max_epoch'))
    parser.add_argument('--learning_rate', type=float, default=config.getfloat('train', 'learning_rate'))
    parser.add_argument('--weight_decay', type=float, default=config.getfloat('train', 'weight_decay'))
    parser.add_argument('--dropout', type=float, default=config.getfloat('train', 'dropout'))
    parser.add_argument('--step_size', type=float, default=config.getfloat('train', 'step_size'))

    parser.add_argument('--input_len', type=int, default=config.getint('data', 'input_len'))
    parser.add_argument('--output_len', type=int, default=config.getint('data', 'output_len'))
    parser.add_argument('--train_ratio', type=float, default=config.getfloat('data', 'train_ratio'))
    parser.add_argument('--val_ratio', type=float, default=config.getfloat('data', 'val_ratio'))
    parser.add_argument('--test_ratio', type=float, default=config.getfloat('data', 'test_ratio'))

    parser.add_argument('--layers', type=int, default=config.getint('param', 'layers'))
    parser.add_argument('--node_num', type=int, default=config.getint('param', 'nodes'))
    parser.add_argument('--tod', type=int, default=config.getint('param', 'tod'))
    parser.add_argument('--dow', type=int, default=config.getint('param', 'dow'))
    parser.add_argument('--input_dims', type=int, default=config.getint('param', 'id'))
    parser.add_argument('--node_dim', type=int, default=config.getint('param', 'nd'))
    parser.add_argument('--tod_dims', type=int, default=config.getint('param', 'td'))
    parser.add_argument('--dow_dims', type=int, default=config.getint('param', 'dd'))

    parser.add_argument('--support_num', type=int, default=config.getint('param', 'support_num'))
    parser.add_argument('--hidden_dim', type=int, default=config.getint('param', 'hidden_dim'))
    parser.add_argument('--cvrelation_dim', type=int, default=config.getint('param', 'cvrelation_dim'))
    parser.add_argument('--cv_fea_dim', type=int, default=config.getint('param', 'cv_fea_dim'))
    parser.add_argument('--cv_token', type=int, default=config.getint('param', 'cv_token'))
    parser.add_argument('--cv_pattern', type=int, default=config.getint('param', 'cv_pattern'))
    parser.add_argument('--relation_patterns', type=int, default=config.getint('param', 'relation_patterns'))
    parser.add_argument('--st_encoder', type=int, default=config.getint('param', 'st_encoder'))
    parser.add_argument('--node_cv_num', type=int, default=config.getint('param', 'node_cv_num'))

    parser.add_argument('--partten_dim', type=int, default=config.getint('param', 'partten_dim'))
    parser.add_argument('--active', type=str, default=config.get('param', 'act'))

    parser.add_argument('--traffic_file', default=config.get('file', 'traffic'))
    parser.add_argument('--meta_file', default=config.get('file', 'meta'))
    parser.add_argument('--adj_file', default=config.get('file', 'adj'))
    parser.add_argument('--model_file', default=config.get('file', 'model'))
    parser.add_argument('--image_file', default=config.get('file', 'image'))
    parser.add_argument('--cvmodel_file', default=config.get('file', 'cvmodel'))
    parser.add_argument('--log_file', default=config.get('file', 'log'))

    args = parser.parse_args()
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_file = args.log_file + "_" + current_time + "/"
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    

    log = open(args.log_file+"/log", 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))

    solver.train()
    solver.test()