import os
import logging
import numpy as np
import pandas
import pickle
import argparse
import random
import shutil
import torch
import torch.nn.functional as F
import baselineUtils

import torch.distributions.multivariate_normal as torchdist
import torch.multiprocessing as multiprocessing

from utils import * 
from metrics import *
from model import *

from contrast.model import *
from contrast.contrastive import *

from transformer.noam_opt import NoamOpt

# random_seed = 2021
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred,V_target)


def train(model, contrastive, optimizer, device, loader_train, epoch, metrics, args):
    #  metrics = {'train_loss': [], 'task_loss': [], 'contrast_loss': [], 'val_loss': []}
    model.train()
    loss_batch, loss_total_batch, loss_contrast_batch = 0, 0, 0
    batch_count = 0

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, safety_gt_ = batch

        obs_traj = obs_traj.type(torch.FloatTensor).to(device)
        pred_traj_gt = pred_traj_gt.type(torch.FloatTensor).to(device)
        V_obs = V_obs.type(torch.FloatTensor).to(device)
        A_obs = A_obs.type(torch.FloatTensor).to(device)
        V_tr = V_tr.type(torch.FloatTensor).to(device)
        A_tr = A_tr.type(torch.FloatTensor).to(device)

        # obs_traj: [1, 64, 2, 8]; pred_traj_gt: [1, 64, 2, 12]
        # [1,8,64,2] [1,8,64,64] [1,12,64,2] [1,12,64,64]
                
        pick_safe_traj = args.safe_traj
        num_person = pred_traj_gt.size(1)
        #lanni pick_safe_traj
        safety_gt = safety_gt_.view(-1) if pick_safe_traj else torch.ones(num_person).bool().to(device) # num_person,
        if pick_safe_traj and safety_gt.sum() == 0:
            # skip this batch if there is no collision-free trajectories
            continue

        # optimizer.zero_grad()
        optimizer.optimizer.zero_grad()
        #Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)  # [1, 2, 8, num_person]  <- [1, 8, num_person, 2]
        A_obs_tmp = A_obs.squeeze() # [2, num_person, num_person]  <- [1, num_person, num_person, 2]
        V_tr_tmp = V_tr.permute(0, 3, 1, 2)  # [1, 2, 12, num_person]  <- [1, 12, num_person, 2]
        # V_tr_tmp_start = torch.zeros(1,2,1,V_tr_tmp.size()[3]).type(torch.FloatTensor).to(device) # [1, 2, 1, num_person]
        V_tr_tmp_start = V_obs_tmp[:,:,-1:,:]
        V_tr_tmp = torch.cat((V_tr_tmp_start, V_tr_tmp[:,:,1:,:]),dim=2) # [1, 2, 12, num_person]
        
        
        V_pred, _, feat_vec = model(V_obs_tmp, A_obs_tmp, V_tr_tmp, return_feat=True)  # [1, 5, 12, num_person], [1, num_person, 60]

        V_pred = V_pred.permute(0, 2, 3, 1)  # [1, 12, num_person, 5] <- [1, 5, 12, num_person]
        feat_vec = feat_vec.squeeze(0)  # [num_person, 60]

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        V_pred = V_pred[:, safety_gt, :].contiguous()
        V_tr = V_tr[:, safety_gt, :].contiguous()
        loss_task = graph_loss(V_pred, V_tr)
        loss_contrast = torch.tensor(0.0).float().to(device)

        # contrastive task

        if args.contrast_weight > 0:
            # Recall dimensionality:
            # obs_traj: [1, num_person, 2, 8]; pred_traj_gt: [1, num_person, 2, 12]

            mask_graph = A_obs_tmp[1,:,:].to(device) # torch.Size([64, 64])
            mask_temp=~torch.eye(64).type(torch.BoolTensor).to(device) # 对角线False
            mask=torch.ones(64,63).to(device)
            mask[:,:]=mask_graph[mask_temp].reshape(64,63).type(torch.BoolTensor)

            # replicate the scene such that each agent is primary for once
            num_person = feat_vec.size(0) # torch.Size([64, 60])
            num_neighbors = num_person - 1

            pedestrain_states = torch.zeros([num_person, 6]).float().to(device)
            pedestrain_states[:, :2] = obs_traj[0, :, :, -1]  # pick input's last frame

            pos_seeds = pred_traj_gt[0, :, :, :args.contrast_horizon].permute(0, 2, 1)  # [num_person, H, 2]

            # trick: swap primary agent for N times, N = num_person
            neg_seeds = torch.zeros([num_person, args.contrast_horizon, num_neighbors, 2]).float().to(device)  # [num_person, H, num_person-1, 2]
            for idx_primary in range(num_person):
                neighbor_idxes = np.delete(np.arange(num_person), idx_primary)
                neg_seeds_tmp = pred_traj_gt[0, np.ix_(neighbor_idxes), :, :args.contrast_horizon].squeeze(0)  # [num_person-1, 2, H]
                neg_seeds[idx_primary] = neg_seeds_tmp.permute(2, 0, 1)  # [H, num_person-1, 2]

            hist_traj = V_obs_tmp.permute(3, 2, 1, 0).reshape(num_person, -1).contiguous()  # [num_person, 16] <- [1, 2, 8, num_person]
            l_contrast = contrastive.loss(pedestrain_states, mask, pos_seeds, neg_seeds, feat_vec, hist_traj)
            # 64*6 64*63 64*4*2 64*4*63*2 64*60 64*16
            loss_contrast += l_contrast * args.contrast_weight


        loss_total = loss_task + loss_contrast
        loss_total.backward()
        
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

        optimizer.step()
        #Metrics
        loss_batch += loss_task.item()
        loss_contrast_batch += loss_contrast.item()
        loss_total_batch += loss_total.item()
        # log error
        # logging.info('TRAIN: Epoch:{:.3f}, Total loss:{:.3f}, task loss:{:.3f},contrast loss:{:.3f}')
        print('TRAIN: Epoch:{:.6f}, Total loss:{:.6f}, task loss:{:.6f},contrast loss:{:.6f}'.format(epoch, loss_total, loss_task, loss_contrast))

    logging.info('######################################################################################')
    logging.info('TRAIN: Epoch:{:.6f}, Total loss:{:.6f}, task loss:{:.6f},contrast loss:{:.6f}'.format(epoch, loss_total_batch/batch_count, loss_batch/batch_count, loss_contrast_batch/batch_count))
    metrics['train_loss'].append(loss_total_batch/batch_count)
    metrics['task_loss'].append(loss_batch/batch_count)
    metrics['contrast_loss'].append(loss_contrast_batch/batch_count)
    

def vald(model, device, loader_val, epoch, metrics, constant_metrics, args):
    model.eval()
    loss_batch = 0
    batch_count = 0

    num_batch = len(loader_val)
    V_pred_rel_to_abs_ksteps_ls, V_y_rel_to_abs_ls, mask_ls = [None] * num_batch, [None] * num_batch, [None] * num_batch

    ade_bigls = []
    fde_bigls = []

    coll_joint_data_bigls = []


    time_start = time.time()
    time_sampling = 0.0


    for cnt,batch in enumerate(loader_val):
        batch_count += 1

        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        # [1,8,64,2] [1,8,64,64] [1,12,64,2] [1,12,64,64]

        V_obs = V_obs.type(torch.FloatTensor).to(device)
        A_obs = A_obs.type(torch.FloatTensor).to(device)
        V_tr = V_tr.type(torch.FloatTensor).to(device) 
        A_tr = A_tr.type(torch.FloatTensor).to(device) 

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        A_obs_tmp = A_obs.squeeze()

        V_tr_tmp = V_tr.permute(0, 3, 1, 2)  # [1, 2, 12, num_person]  <- [1, 12, num_person, 2]
        # V_tr_tmp_start = torch.zeros(1,2,1,V_tr_tmp.size()[3]).type(torch.FloatTensor).to(device) # [1, 2, 1, num_person]
        V_tr_tmp_start = V_obs_tmp[:,:,-1:,:]
        V_tr_tmp = torch.cat((V_tr_tmp_start, V_tr_tmp[:,:,1:,:]),dim=2) # [1, 2, 12, num_person]
        
        
        V_pred, _= model(V_obs_tmp, A_obs_tmp, V_tr_tmp)  # [1, 5, 12, num_person], [1, num_person, 60]

        # V_pred,_ = model(V_obs_tmp, A_obs_temp)

        V_pred = V_pred.permute(0, 2, 3, 1) #  [1, 12, num_person, 5]]

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()


        loss_task = graph_loss(V_pred,V_tr)


        #Metrics
        loss_batch += loss_task.item()
        # log error
        print('VALD: Epoch:{:.6f},Loss:{:.6f}'.format(epoch,loss_task))
        # logging.info('Time to multiprocess all {:d} pieces of batch data: {:.3f}s'.format(num_batch, time_elapsed))


        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # For now I have my bi-variate parameters
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]
        # dimensionality reminder: mean: [12, num_person, 2], cov: [12, num_person, 2, 2]

        """pytorch solution for sampling"""
        time_sampling_start = time.time()

        mvnormal = torchdist.MultivariateNormal(mean, cov)
        kstep_V_pred_ls = []
        KSTEPS=20
        for i in range(KSTEPS):
            kstep_V_pred_ls.append(mvnormal.sample().cpu().numpy())  # cat [12, num_person, 2]
        kstep_V_pred_ls = np.stack(kstep_V_pred_ls, axis=0) # [KSTEPS, 12, num_person, 2]

        kstep_V_pred = np.concatenate([traj for traj in kstep_V_pred_ls], axis=1) # [12, KSTEPS * num_person, 2]

        time_sampling_elapsed = time.time() - time_sampling_start
        time_sampling += time_sampling_elapsed
        """end of sampling"""

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy()) # [8, num_person, 2]
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :]) # [12, num_person, 2] speed???

        kstep_V_x = np.concatenate([V_x[-1, :, :]] * KSTEPS, axis=0)  # cat along number of person
        kstep_V_pred_rel_to_abs = nodes_rel_to_nodes_abs(kstep_V_pred, kstep_V_x).reshape(12, KSTEPS, num_of_objs, 2)
        kstep_V_pred_rel_to_abs = kstep_V_pred_rel_to_abs.transpose((1, 0, 2, 3))  # [KSTEPS, 12, num_object, 2]

        V_pred_rel_to_abs_ksteps_ls[cnt] = kstep_V_pred_rel_to_abs  # np.ndarray
        V_y_rel_to_abs_ls[cnt] = V_y_rel_to_abs  # np.ndarray
        mask_ls[cnt] =  A_obs_tmp[1,:,:].cpu().numpy()!=0 #cnt*64*64
    
    time_elapsed = time.time() - time_start

    logging.info('###########################################')
    logging.info('VALD: Epoch:{:.6f},Loss:{:.6f}'.format(epoch,loss_batch/batch_count))
    metrics['val_loss'].append(loss_batch/batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
    logging.info('VALD: Best Epoch:{:.6f}, Best Loss:{:.6f}'.format(constant_metrics['min_val_epoch'],constant_metrics['min_val_loss']))

    time_start = time.time()
    func_batch_input = []
    for batch_idx in range(num_batch):
        V_pred_rel_to_abs_ksteps = V_pred_rel_to_abs_ksteps_ls[batch_idx]
        V_y_rel_to_abs = V_y_rel_to_abs_ls[batch_idx]
        mask_pred = mask_ls[batch_idx]
        if epoch == 0:
            cur_tuple = (batch_idx, V_pred_rel_to_abs_ksteps, V_y_rel_to_abs, mask_pred, True)
        else:
            cur_tuple = (batch_idx, V_pred_rel_to_abs_ksteps, V_y_rel_to_abs, mask_pred, False)
        func_batch_input.append(cur_tuple)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_batch_data, func_batch_input)
    time_elapsed = time.time() - time_start
    logging.info('Time to multiprocess all {:d} pieces of batch data: {:.6f}s'.format(num_batch, time_elapsed))

    for idx_proc, result in enumerate(results):

        ade_bigls += result[0]  # list cat
        fde_bigls += result[1]  # list cat
        coll_joint_data_bigls.append(result[3])  # append np.ndarray

    coll_joint_step, coll_joint_cum = coll_data_post_processing(coll_joint_data_bigls)


    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    
    logging.info("VALD: ADE: {:.4f}, FDE: {:.4f}, COL: {:.4f}".format(ade_, fde_, coll_joint_cum[2]))

def coll_data_post_processing(coll_data_bigls):
    coll_raw_ = np.concatenate(coll_data_bigls, axis=0)  # [X, 56]
    coll_step_ = np.mean(coll_raw_, axis=0)  # [56]
    coll_step_ = coll_step_[:-1].reshape(-1, 5).mean(axis=1)  # [11]
    coll_cumulative_ = np.asarray([np.mean(coll_raw_[:, :i * 5 + 6].max(axis=1)) for i in range(11)])  # int
    return coll_step_, coll_cumulative_

def stack_dict(data_dict):
    for key, coll_step_data in zip(data_dict.keys(), data_dict.values()):
        data_dict[key] = np.stack(coll_step_data, axis=0)  # [X, 56]
    return data_dict


def process_batch_data(batch_idx: int, V_pred_rel_to_abs_ksteps: np.ndarray, V_y_rel_to_abs: np.ndarray, mask_pred: np.ndarray, compute_col_truth=False):
     # [KSTEPS, 12, num_object, 2]  # [12, num_object, 2]
    ade_ls = {}
    fde_ls = {}
    coll_ls = {}
    coll_joint_data_ls = {}
    coll_cross_data_ls = {}
    coll_truth_data_ls = {}

    num_of_objs = V_y_rel_to_abs.shape[1]
    for n in range(num_of_objs):
        ade_ls[n] = []
        fde_ls[n] = []
        coll_ls[n] = []
        coll_joint_data_ls[n] = []
        coll_cross_data_ls[n] = []
        coll_truth_data_ls[n] = []

    KSTEPS = len(V_pred_rel_to_abs_ksteps)
    # print('Detected ksteps: {:d}'.format(KSTEPS))
    for k in range(KSTEPS):
        V_pred_rel_to_abs = V_pred_rel_to_abs_ksteps[k]

        for n in range(num_of_objs):
            pred = [V_pred_rel_to_abs[:, n:n + 1, :]] # 12,1,2
            target = [V_y_rel_to_abs[:, n:n + 1, :]] # 12,1,2
            number_of = [1]

            ade_ls[n].append(ade(pred, target, number_of))
            fde_ls[n].append(fde(pred, target, number_of))

            ######
            predicted_traj = V_pred_rel_to_abs[:, n, :]  # [12, 2]
            predicted_trajs_all = V_pred_rel_to_abs.transpose(1, 0, 2)  # [num_person, 12, 2]
            mask = mask_pred[n,:] # 64
            col_mask_joint = compute_col_pred(predicted_traj, predicted_trajs_all, mask).astype(np.float64)  # [56], between predictions

            target_traj = V_y_rel_to_abs[:, n, :]  # [12, 2]
            target_trajs_all = V_y_rel_to_abs.transpose(1, 0, 2)  # [num_person, 12, 2]
            col_mask_cross = compute_col_pred(predicted_traj, target_trajs_all, mask).astype(np.float64)  # [56], prediction x ground-truth

            if compute_col_truth: # fist epoch
                col_mask_truth = compute_col_pred(target_traj, target_trajs_all, mask).astype(np.float64)  # [56], between ground-truth
                coll_truth_data_ls[n].append(col_mask_truth)

            if col_mask_joint.sum():
                coll_ls[n].append(1)
            else:
                coll_ls[n].append(0)
            coll_joint_data_ls[n].append(col_mask_joint) # object*20*56
            coll_cross_data_ls[n].append(col_mask_cross)
            ######

    coll_joint_data_ls = stack_dict(coll_joint_data_ls)  # object*20*56
    coll_cross_data_ls = stack_dict(coll_cross_data_ls)
    if compute_col_truth:
        coll_truth_data_ls = stack_dict(coll_truth_data_ls)
    #  internal processing ends

    #  write data to the returned list, appending is okay as the order is not important
    ade_bigls_item, fde_bigls_item, coll_bigls_item = [], [], []
    for n in range(num_of_objs):
        ade_bigls_item.append(min(ade_ls[n]))  # float
        fde_bigls_item.append(min(fde_ls[n]))  # float
        coll_bigls_item.append(sum(coll_ls[n]) / len(coll_ls[n]))  # float
    coll_joint_data_bigls_item = np.concatenate([ls for ls in coll_joint_data_ls.values()], axis=0)  # [object*20, 56], np.ndarray
    coll_cross_data_bigls_item = np.concatenate([ls for ls in coll_cross_data_ls.values()], axis=0)
    if compute_col_truth:
        coll_truth_data_bigls_item = np.concatenate([ls for ls in coll_truth_data_ls.values()], axis=0)
    else:
        coll_truth_data_bigls_item = None

    return ade_bigls_item, fde_bigls_item, coll_bigls_item, coll_joint_data_bigls_item, coll_cross_data_bigls_item, coll_truth_data_bigls_item


def sample_pred(V_pred, V_tr, i):
    # V_tr [1,12,64,2]
    device= V_pred.device
    V_pred = V_pred.permute(0, 2, 3, 1) #  [1, 12, num_person, 5]]
    V_pred = V_pred[:,-1:,:,:]
    V_pred = V_pred.squeeze(0) #  [-1, num_person, 5]
    # For now I have my bi-variate parameters
    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = V_pred[:, :, 0:2]

    mvnormal = torchdist.MultivariateNormal(mean, cov)


    # V_pred_result = mvnormal.sample().reshape(1,-1,2)
    kstep_V_pred_ls = []
    for j in range(20):
        kstep_V_pred_ls.append(mvnormal.sample())  # cat [-1, num_person, 2]
    kstep_V_pred = torch.cat(kstep_V_pred_ls,dim=0) #[-1*20, num_person, 2]
    
    V_this = V_tr.squeeze()[i:i+1,:,:] #[-1, num_person, 2]

    distance = F.pairwise_distance(kstep_V_pred.reshape(-1,2), V_this.repeat(20,1,1).reshape(-1,2), p=2).reshape(-1,kstep_V_pred.size()[1])
    index=torch.argmin(distance,dim=0)
    index=index.reshape(1,kstep_V_pred.size()[1],1).repeat(1,1,2)
    V_pred_result=torch.gather(kstep_V_pred, 0, index) #[-1, num_person, 2]

    return V_pred_result


def test(model, device, loader_test, epoch, KSTEPS=20):
    model.eval()
    loss_batch = 0
    batch_count = 0
    # save batch data to list for later multi-processing
    num_batch = len(loader_test)
    V_pred_rel_to_abs_ksteps_ls, V_y_rel_to_abs_ls, mask_ls = [None] * num_batch, [None] * num_batch, [None] * num_batch

    ade_bigls = []
    fde_bigls = []
    coll_bigls = []
    coll_joint_data_bigls = []
    coll_cross_data_bigls = []
    coll_truth_data_bigls = []
    raw_data_dict = {}

    time_start = time.time()
    time_sampling = 0.0

    for step, batch in enumerate(loader_test):
        batch_count += 1
        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch
         # [1,8,64,2] [1,8,64,64] [1,12,64,2] [1,12,64,64]
        obs_traj = obs_traj.type(torch.FloatTensor).to(device)
        obs_traj_rel = obs_traj_rel.type(torch.FloatTensor).to(device)
        V_obs = V_obs.type(torch.FloatTensor).to(device)
        A_obs = A_obs.type(torch.FloatTensor).to(device)
        V_tr = V_tr.type(torch.FloatTensor).to(device) 
 
        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        A_obs_tmp = A_obs.squeeze()

        # V_tr_tmp_start = torch.zeros(1,2,1,V_tr.size()[2]).type(torch.FloatTensor).to(device) # [1, 2, 1, num_person]
        V_tr_tmp_start = V_obs_tmp[:,:,-1:,:]
        V_tr_tmp = V_tr_tmp_start # [1, 2, 1, num_person]


        for i in range(V_tr.shape[1]):
            V_pred , _  = model(V_obs_tmp, A_obs_tmp, V_tr_tmp) #  [1, 5, 1, num_person]
            output=  sample_pred(V_pred, V_tr, i) #  [-1, num_person, 2]
            output = output.permute(2,0,1).unsqueeze(0)
            V_tr_tmp = torch.cat((V_tr_tmp, output), 2)

        # V_pred,_ = model(V_obs_tmp, A_obs_temp)

        V_pred = V_pred.permute(0, 2, 3, 1) #  [1, 12, num_person, 5]]

        # V_pred, _ = model(V_obs_tmp, A_obs_tmp) #  [1, 5, 12, num_person]]
        # V_pred = V_pred.detach().permute(0, 2, 3, 1) #  [1, 12, num_person, 5]]

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()  #  [12, num_person, 5]]

        loss_task = graph_loss(V_pred,V_tr)
        loss_batch += loss_task.item()

        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # For now I have my bi-variate parameters
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]
        # dimensionality reminder: mean: [12, num_person, 2], cov: [12, num_person, 2, 2]

        """pytorch solution for sampling"""
        time_sampling_start = time.time()

        mvnormal = torchdist.MultivariateNormal(mean, cov)
        kstep_V_pred_ls = []
        for i in range(KSTEPS):
            kstep_V_pred_ls.append(mvnormal.sample().cpu().numpy())  # cat [12, num_person, 2]
        kstep_V_pred_ls = np.stack(kstep_V_pred_ls, axis=0) # [KSTEPS, 12, num_person, 2]

        kstep_V_pred = np.concatenate([traj for traj in kstep_V_pred_ls], axis=1) # [12, KSTEPS * num_person, 2]

        time_sampling_elapsed = time.time() - time_sampling_start
        time_sampling += time_sampling_elapsed
        """end of sampling"""

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy()) # [8, num_person, 2]
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :]) # [12, num_person, 2] speed???

        kstep_V_x = np.concatenate([V_x[-1, :, :]] * KSTEPS, axis=0)  # cat along number of person
        kstep_V_pred_rel_to_abs = nodes_rel_to_nodes_abs(kstep_V_pred, kstep_V_x).reshape(12, KSTEPS, num_of_objs, 2)
        kstep_V_pred_rel_to_abs = kstep_V_pred_rel_to_abs.transpose((1, 0, 2, 3))  # [KSTEPS, 12, num_object, 2]

        V_pred_rel_to_abs_ksteps_ls[step] = kstep_V_pred_rel_to_abs  # np.ndarray
        V_y_rel_to_abs_ls[step] = V_y_rel_to_abs  # np.ndarray
        mask_ls[step] =  A_obs_tmp[1,:,:].cpu().numpy()!=0 #cnt*64*64
    
    time_elapsed = time.time() - time_start
    # log error
    logging.info('TEST: Epoch:{:.6f},Loss:{:.6f}'.format(epoch,loss_batch/batch_count))
    logging.info('Time to prepare all {:d} pieces of batch data: {:.6f}s'.format(num_batch, time_elapsed))
    logging.info('In particular, time for multivariate gaussian distribution sampling: {:.6f}s'.format(time_sampling))

    time_start = time.time()
    func_batch_input = []
    for batch_idx in range(num_batch):
        V_pred_rel_to_abs_ksteps = V_pred_rel_to_abs_ksteps_ls[batch_idx]
        V_y_rel_to_abs = V_y_rel_to_abs_ls[batch_idx]
        mask_pred = mask_ls[batch_idx]
        if epoch == 0:
            cur_tuple = (batch_idx, V_pred_rel_to_abs_ksteps, V_y_rel_to_abs, mask_pred, True)
        else:
            cur_tuple = (batch_idx, V_pred_rel_to_abs_ksteps, V_y_rel_to_abs, mask_pred, False)
        func_batch_input.append(cur_tuple)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_batch_data, func_batch_input)
    time_elapsed = time.time() - time_start
    logging.info('Time to multiprocess all {:d} pieces of batch data: {:.6f}s'.format(num_batch, time_elapsed))

    for idx_proc, result in enumerate(results):
        ade_bigls += result[0]  # list cat
        fde_bigls += result[1]  # list cat
        coll_bigls += result[2]  # list cat
        coll_joint_data_bigls.append(result[3])  # append np.ndarray
        coll_cross_data_bigls.append(result[4])
        if epoch == 0:
            coll_truth_data_bigls.append(result[5])  # could be None

    coll_joint_step, coll_joint_cum = coll_data_post_processing(coll_joint_data_bigls)
    coll_cross_step, coll_cross_cum = coll_data_post_processing(coll_cross_data_bigls)
    if epoch == 0:
        coll_truth_step, coll_truth_cum = coll_data_post_processing(coll_truth_data_bigls)
    else:
        coll_truth_step, coll_truth_cum = None, None

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    coll_ = sum(coll_bigls) / len(coll_bigls)

    return ade_, fde_, coll_, coll_joint_step, coll_joint_cum, coll_cross_step, coll_cross_cum, coll_truth_step, coll_truth_cum, raw_data_dict

def config_parser():
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_sstgcn', type=int, default=3)
    parser.add_argument('--n_txpcnn', type=int, default=5)
    parser.add_argument('--kernel_size', type=int, default=3)

    # Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='zara1test',
                        help='eth,hotel,univ,zara1,zara2')

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=500, # 500
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--folder', default='checkpoint', # 
                        help='personal folder for the model ')
    parser.add_argument('--tag', default='tag', #
                        help='personal tag for the model ')

    # ------------------contrast setting-------------------------------

    parser.add_argument('--contrast_sampling', type=str, default='event') #
    parser.add_argument('--contrast_weight', type=float, default=0.05) # 0.05
    parser.add_argument('--contrast_horizon', type=int, default=4)
    parser.add_argument('--contrast_temperature', type=float, default=0.2)
    parser.add_argument('--contrast_range', type=float, default=2.0)
    parser.add_argument('--contrast_nboundary', type=int, default=0)
    parser.add_argument('--ratio_boundary', type=float, default=0.5)
    parser.add_argument('--contrast_loss', type=str, default='nce')
    parser.add_argument('--contrast_minsep', type=float, default=0.2)
    parser.add_argument('--safe_traj', action='store_true', default=False,
                        help='remove training trajectories with collision')

    # ------------------transformer setting-------------------------------

    parser.add_argument('--emb_size',type=int,default=32)
    parser.add_argument('--heads',type=int, default=4)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--fw',type=int, default=128)


    args = parser.parse_args()
    return args


def get_target_metrics(dataset: str, tolerance: float = 0.0):

    if dataset == 'eth':
        # target_ade, target_fde = 0.64, 1.11  # paper
        target_ade, target_fde = 0.732, 1.223  #
        target_col = 1.33
    elif dataset == 'hotel':
        # target_ade, target_fde = 0.49, 0.85  # paper
        target_ade, target_fde = 0.410, 0.671  # 
        target_col = 3.56
    elif dataset == 'univ':
        # target_ade, target_fde = 0.44, 0.79  # paper
        target_ade, target_fde = 0.489, 0.911  # 
        target_col = 9.22
    elif dataset == 'zara1':
        target_ade, target_fde = 0.335, 0.524  # paper 
        target_col = 2.14
    elif dataset == 'zara2':
        target_ade, target_fde = 0.304, 0.481  # paper 
        target_col = 6.87
    else:
        raise NotImplementedError
    return target_ade+tolerance, target_fde+tolerance, target_col


def config_model(args, device, checkpoint_dir):
    """Define the model."""
    '''
    model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)
    '''               

    model = SGTN(n_sstgcn=args.n_sstgcn, n_txpcnn=args.n_txpcnn,
                          output_feat=args.output_size, seq_len=args.obs_seq_len,
                          kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len,
                          emb_size=args.emb_size, fw=args.fw, heads=args.heads,layers=args.layers,dropout=args.dropout,
                          checkpoint_dir=checkpoint_dir).to(device)

    projection_head = ProjHead(feat_dim=args.pred_seq_len*5 + (args.obs_seq_len)*2, hidden_dim=32, head_dim=8).to(device) # 60+16 
    if args.contrast_sampling == 'event':
        encoder_sample = EventEncoder(hidden_dim=8, head_dim=8).to(device)
    else:
        encoder_sample = SpatialEncoder(hidden_dim=8, head_dim=8).to(device)
    num_params_contrast = sum(
        [p.numel() for layer in [projection_head, encoder_sample] for p in layer.parameters() if p.requires_grad])
    logging.info('Contrastive learning module # trainable parameters: {:d}'.format(num_params_contrast))

    # contrastive
    if args.contrast_loss == 'nce':
        contrastive = SocialNCE(projection_head, encoder_sample, args.contrast_sampling, args.contrast_horizon,
                                args.contrast_nboundary, args.contrast_temperature, args.contrast_range,
                                args.ratio_boundary, args.contrast_minsep)
    else:
        raise NotImplementedError
    return model, contrastive


def get_dataloader(bs, dataset, obs_seq_len, pred_seq_len, checkpoint_dir):
    data_set = '../../../scratch/data/SGTN/datasets/' + dataset + '/'

    dset_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True,
        checkpoint_dir=checkpoint_dir)

    loader_train = DataLoader(
        dset_train,
        batch_size=bs,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=6, pin_memory=True)

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True,
        checkpoint_dir=checkpoint_dir)

    loader_val = DataLoader(
        dset_val,
        batch_size=bs,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=6, pin_memory=True)

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True,
        checkpoint_dir=checkpoint_dir)

    loader_test = DataLoader(
        dset_test,
        batch_size=bs,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=6, pin_memory=True)



    return loader_train, loader_val, loader_test #, mean, std


def pick_from_log(args, log_path: str, min_epoch: int = 50):
    """Read training log from checkpoint folder."""
    log_name = '-'.join(os.path.basename(os.path.dirname(log_path)).split('-')[:-3])
    dataset = args.dataset
    # os.path.basename(os.path.abspath(os.path.join(log_path, '..'))).split('-')[-1]
    if not os.path.exists(log_path):
        logging.info('Expected training log at {:s} does not exist.'.format(log_path))
        return None
    model_weights = [anything for anything in os.listdir(os.path.join(os.path.dirname(log_path), 'history')) if anything.endswith('best.pth')]
    if len(model_weights) < min_epoch:
        logging.info('Training epochs {:d} are too few!'.format(len(model_weights)))
        return None
    else:
        df_raw = pandas.read_csv(log_path)
        if 'col_joint_c4' in df_raw.columns:
            columns_to_pick = ['Epoch', 'ADE', 'FDE', 'col_joint_c4']
        else:
            columns_to_pick = ['Epoch', 'ADE', 'FDE', 'COLL']
        df_ = df_raw[columns_to_pick]

        _, target_fde, _ = get_target_metrics(dataset, 0.001)
        best_fde_overall = df_['FDE'].values.min()
        if best_fde_overall > target_fde:
            col_joint_c4_error = df_['ADE'].values + df_['FDE'].values
            best_col_idx = np.argsort(col_joint_c4_error)[0]
            best_col_epoch = int(df_['Epoch'].values[best_col_idx])
            best_col_ade = df_['ADE'].values[best_col_idx]
            best_col_fde = df_['FDE'].values[best_col_idx]
            best_col = df_['col_joint_c4'][best_col_idx] if 'col_joint_c4' in df_raw.columns else df_['COLL'][best_col_idx]
            logging.info('####---NO--- best_fde_overall > target_fde ---- ####')
            logging.info('ADE+FDE+COL total error minimizer: ADE: {:.6f}, FDE: {:.6f}, COL: {:.6f}%, EPOCH: {:d}.'.format(
                best_col_ade, best_col_fde, best_col * 100, best_col_epoch))
            return best_col_epoch

        tolerance_ls = [0.001]
        for tolerance in tolerance_ls:
            # find most performant model by ADE/FDE tolerance
            target_ade, target_fde, target_col = get_target_metrics(dataset, tolerance)
            mask_good_fde = df_['FDE'].values <= target_fde
            df = df_.loc[mask_good_fde]

            if mask_good_fde.sum() == 0:
                continue

            best_fde = df['FDE'].values.min()
            if best_fde > target_fde:
                logging.info('Tolerance: {:.6f}, FDE too large: {:.6f} > target = {:.6f}'.format(tolerance, best_fde, target_fde))
                return None
            else:
                coll_overall = df['col_joint_c4'].values if 'col_joint_c4' in df.columns else df['COLL'].values
                best_col = coll_overall.min()
                best_col_idx = np.argsort(coll_overall)[0]
                best_col_epoch = int(df['Epoch'].values[best_col_idx])
                best_col_ade = df['ADE'].values[best_col_idx]
                best_col_fde = df['FDE'].values[best_col_idx]
                logging.info('####---YES--- best_fde < target_fde ---- ####')
                logging.info('Tolerance: {:.6f}, Best FDE: {:.6f} <= target = {:.6f} '.format(tolerance, best_fde, target_fde))
                logging.info('Best model up to now: ADE: {:.6f}, FDE: {:.6f}, COL: {:.6f}%, EPOCH: {:d}'.format(
                    best_col_ade, best_col_fde, best_col * 100, best_col_epoch))
        return best_col_epoch


def main():

    # 参数
    args = config_parser()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    target_ade, target_fde, target_col = get_target_metrics(args.dataset)
    # to be very conservative
    target_ade -= 0.05
    target_fde -= 0.05

     # Training log settings
    checkpoint_dir = '../../../scratch/experiment/' + args.folder + '/' + args.tag + '/'
    history_dir = os.path.join(checkpoint_dir, 'history') + '/'
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')

    for folder in [checkpoint_dir, history_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    set_logger(os.path.join(checkpoint_dir, "train.log"))


    logging.info('*' * 30)
    logging.info("Training initiating....")
    logging.info(args)

    # Define the model
    # lanni:Contrastive learning module # trainable parameters: 2976
    model, contrastive = config_model(args, device, checkpoint_dir)

    # Data loader
    loader_train, loader_val, loader_test = get_dataloader(args.batch_size, args.dataset, args.obs_seq_len, args.pred_seq_len, checkpoint_dir)

    # Optimizer settings
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # transformer optimizer
    optimizer = NoamOpt(args.emb_size, args.factor, len(loader_train)*args.warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    if args.use_lrschd:
        patience_epoch = args.lr_sh_rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epoch, threshold=0.01,
                                                         factor=0.5, cooldown=patience_epoch, min_lr=1e-5, verbose=True)


    # save argument once and for all
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    logging.info('Checkpoint dir:{:s}' .format(checkpoint_dir))

    metrics = {'train_loss': [], 'task_loss': [], 'contrast_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    # Start training
    logging.info('Training started ...')
    ade_ls, fde_ls, coll_ls, ttl_error_ls = [], [], [], []
    best_ade, best_fde, best_coll, best_ttl_error, best_coll_joint_c4_error, best_coll_joint_c4 = 99999., 99999., 99999., 99999., 99999., 99999.

    df = pandas.DataFrame(columns=['Epoch', 'total_loss', 'task_loss', 'contrast_loss', 'validation_loss', 'ADE', 'FDE', 'COLL'])
    
    for epoch in range(args.num_epochs):
        time_start = time.time()
        train(model, contrastive, optimizer, device, loader_train, epoch, metrics, args)
        time_elapsed = time.time() - time_start
        logging.info('Time to train once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))

        time_start = time.time()
        vald(model, device, loader_val, epoch, metrics, constant_metrics, args)
        time_elapsed = time.time() - time_start
        logging.info('Time to validate once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))
        if args.use_lrschd:
            ttl_loss = metrics['train_loss'][-1]
            scheduler.step(ttl_loss)  # learning rate decay once training stagnates

        logging.info('###########################################')
        logging.info('Epoch:{:s} : {:d}'.format(args.tag,epoch))
        for k, v in metrics.items():
            if len(v) > 0:
                logging.info('{:s}: {:.6f}'.format(k, v[-1]))

        """Test per epoch"""
        ade_, fde_, coll_ = 999999.0, 999999.0, 999999.0
        logging.info("Testing ....")
        time_start = time.time()
        ad, fd, coll, coll_joint_step, coll_joint_cum, coll_cross_step, coll_cross_cum, coll_truth_step, coll_truth_cum, _ = test(
            model, device, loader_test, epoch)        
        
        # lanni: coll_joint_cum
        time_elapsed = time.time() - time_start
        logging.info('Time to test once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))
        ade_, fde_, coll_ = min(ade_, ad), min(fde_, fd), min(coll_, coll_joint_cum[2])
        ttl_error_ = np.clip(ade_ - target_ade, a_min=0.0, a_max=None) + np.clip(fde_ - target_fde, a_min=0.0, a_max=None) + coll_

        ade_ls.append(ade_)
        fde_ls.append(fde_)
        coll_ls.append(coll_)
        ttl_error_ls.append(ttl_error_)
        logging.info("ADE: {:.4f}, FDE: {:.4f}, COL: {:.4f}, Total ERROR: {:.4f}, COL_JOINT_C4: {:.4F}".format(
            ade_, fde_, coll_, ttl_error_, coll_joint_cum[2]))

        best_ade = min(ade_, best_ade)
        best_fde = min(fde_, best_fde)
        best_coll = min(coll_, best_coll)
        best_ttl_error = min(ttl_error_, best_ttl_error) #加权total
        best_coll_joint_c4 = min(coll_joint_cum[2], best_coll_joint_c4)
        logging.info(
            "Best ADE: {:.4f}, Best FDE: {:.4f}, Best COL: {:.4f}, Best Total ERROR: {:.4f}, Best COL_JOINT_C4: {:.4F}".format(
                best_ade, best_fde, best_coll, best_ttl_error, best_coll_joint_c4))

        df.loc[len(df)] = [epoch, metrics['train_loss'][-1], metrics['task_loss'][-1], metrics['contrast_loss'][-1],
                           metrics['val_loss'][-1], ade_, fde_, coll_]
        df = df.sort_values(by=['Epoch'])
        if not os.path.exists(csv_path):
            df.iloc[-1:].to_csv(csv_path, mode='a', index=False)
        else:
            df.iloc[-1:].to_csv(csv_path, mode='a', header=False, index=False)

        best_epoch = pick_from_log(args, csv_path, 0)
        logging.info('Best epoch up to now is {}'.format(best_epoch))
        """Test ends"""

        logging.info(constant_metrics)
        logging.info('*'*30)

        with open(history_dir+'epoch{:03d}_metrics.pkl'.format(epoch), 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(history_dir+'epoch{:03d}_constant_metrics.pkl'.format(epoch), 'wb') as fp:
            pickle.dump(constant_metrics, fp)

        torch.save(model.state_dict(), history_dir + 'epoch{:03d}_val_best.pth'.format(epoch))

        # model selection
        shutil.copy(history_dir+'epoch{:03d}_metrics.pkl'.format(best_epoch), checkpoint_dir + 'metrics.pkl')
        shutil.copy(history_dir+'epoch{:03d}_constant_metrics.pkl'.format(best_epoch), checkpoint_dir + 'constant_metrics.pkl')
        shutil.copy(history_dir+'epoch{:03d}_val_best.pth'.format(best_epoch), checkpoint_dir + 'val_best.pth')
        

if __name__ == '__main__':
    main()
