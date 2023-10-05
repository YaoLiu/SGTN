import os
import sys
import logging
import numpy as np
import pickle
import argparse
import shutil
import random
import torch
import time
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
from tqdm.auto import tqdm

from utils import * 
from metrics import *
from model import *

from transformer.noam_opt import NoamOpt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_date',type=str, default='0909')
    parser.add_argument('--dataset',type=str, default='zara1test')
    parser.add_argument('--exp',type=str, default='demo1')

    parser.add_argument('--gpu_deterministic', type=bool, default=False)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument('--seed', type=int, default=113)

    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--delim',type=str, default='tab')
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--data_use',type=str, default='graph_data_64.dat')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--KSTEPS',type=int, default=20)

    parser.add_argument('--modelType',type=int, default=1) # 
    parser.add_argument('--attnNei',type=int, default=0) # 
    parser.add_argument('--newTrans',type=int, default=0) # 

    parser.add_argument('--num_sstgcn', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)

    parser.add_argument('--emb_size',type=int,default=8)
    parser.add_argument('--heads',type=int, default=4)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--fw',type=int, default=32)

    args = parser.parse_args()
    return args


def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))

def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_output_dir(folder,dataset,exp):
    output_dir = os.path.join('/home/yaoliu/scratch/experiment/SGTN/' + folder, dataset, exp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def get_dataloader(params, logger):
    data_set = '../../../scratch/data/SGTN/datasets/' + params.dataset + '/'

    dset_train = TrajectoryDataset(
        data_dir=data_set + 'train/',
        logger=logger,
        obs_len=params.obs_seq_len,
        pred_len=params.pred_seq_len,
        skip=params.skip,
        delim=params.delim,
        k=params.k,
        data_use=params.data_use)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,  
        shuffle=True,
        num_workers=6, pin_memory=True)
    
    dset_val = TrajectoryDataset(
        data_dir=data_set + 'val/',
        logger=logger,
        obs_len=params.obs_seq_len,
        pred_len=params.pred_seq_len,
        skip=params.skip,
        delim=params.delim,
        k=params.k,
        data_use=params.data_use)
    
    loader_val = DataLoader(
        dset_val,
        batch_size=1,  
        shuffle=True,
        num_workers=6, pin_memory=True)

    dset_test = TrajectoryDataset(
        data_dir=data_set + 'test/',
        logger=logger,
        obs_len=params.obs_seq_len,
        pred_len=params.pred_seq_len,
        skip=params.skip,
        delim=params.delim,
        k=params.k,
        data_use=params.data_use)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  
        shuffle=False,
        num_workers=6, pin_memory=True)


    return loader_train, loader_val, loader_test


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred,V_target)


def train(model, optimizer, device, loader_train, args):
    model.train()
    loss_total_batch = 0
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

        optimizer.optimizer.zero_grad()
        #Forward

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)  # [1, 2, 8, num_person]  <- [1, 8, num_person, 2]
        A_obs_tmp = A_obs.squeeze() # [2, num_person, num_person]  <- [1, num_person, num_person, 2]
        V_tr_tmp = V_tr.permute(0, 3, 1, 2)  # [1, 2, 12, num_person]  <- [1, 12, num_person, 2]
       
        V_tr_tmp_start = V_obs_tmp[:,:,-1:,:]
        V_tr_tmp = torch.cat((V_tr_tmp_start, V_tr_tmp[:,:,:-1,:]),dim=2) # [1, 2, 12, num_person]
        
        V_pred= model(V_obs_tmp, A_obs_tmp, V_tr_tmp, -1, V_tr_tmp)  # [1, 5, 12, num_person], [1, num_person, 60]

        V_pred = V_pred.permute(0, 2, 3, 1)  # [1, 12, num_person, 5] <- [1, 5, 12, num_person]

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        loss_total = graph_loss(V_pred, V_tr)

        loss_total.backward()

        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

        optimizer.step()
        loss_total_batch += loss_total.item()

    batch_loss = loss_total_batch/batch_count
    return batch_loss


def val_test(model, device, loader, KSTEPS=20):
    model.eval()
    loss_batch = 0
    batch_count = 0
    num_batch = len(loader)

    disbiglist = []

    for step, batch in enumerate(loader):
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

        V_tr_tmp_start = V_obs_tmp[:,:,-1:,:]
        V_tr_tmp = V_tr_tmp_start # [1, 2, 1, num_person]

        V_pred = model(V_obs_tmp, A_obs_tmp, V_tr_tmp, 0, V_tr_tmp)
        for i in range(1,V_tr.shape[1]):
            V_pred = model(V_obs_tmp, A_obs_tmp, V_tr_tmp, i, V_pred) #  [1, 5, 1, num_person]

        V_pred = V_pred.permute(0, 2, 3, 1) #  [1, 12, num_person, 5]]

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()  #  [12, num_person, 5]]

        loss_task = graph_loss(V_pred,V_tr)
        loss_batch += loss_task.item()

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
        kstep_V_pred_ls = []
        for i in range(KSTEPS):
            kstep_V_pred_ls.append(mvnormal.sample().cpu().numpy())  # cat [12, num_person, 2]
        kstep_V_pred_ls = np.stack(kstep_V_pred_ls, axis=0) # [KSTEPS, 12, num_person, 2]


        start = seq_to_nodes(obs_traj.data.cpu().numpy())[-1, :, :] # [8, num_person, 2]
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze(), start) # [12, num_person, 2] torch.Size([25, 64, 2]) torch.Size([64, 2])
        V_y_rel_to_abs =V_y_rel_to_abs.repeat(KSTEPS,1,1,1) #torch.Size([20, 25, 64, 2])
        dislist = [] 

        for i in range(V_y_rel_to_abs.size()[1]):
            thispredrel = torch.tensor(kstep_V_pred_ls[:,i,:,:]).to(device)
            thispred = thispredrel +torch.tensor(start).to(device)
            thisgt = torch.tensor(V_y_rel_to_abs[:,i,:,:]).to(device)
            distance = F.pairwise_distance(thispred, thisgt, p=2)
            index=torch.argmin(distance,dim=0)
            indexNext=index.reshape(1,index.size()[0],1).repeat(1,1,2)
            next=torch.gather(thispred, 0, indexNext) #[-1, num_person, 2]
            start=next.squeeze()
            indexDis=index.reshape(1,index.size()[0])
            disres=torch.gather(distance, 0, indexDis).squeeze() #[-1, num_person, 2]
            disres = disres.sum()/len(disres)
            dislist.append(disres)
        disbiglist.append(dislist)

    final=np.sum(disbiglist,axis=0)/num_batch
    loss=loss_batch/batch_count
    return loss, final

def sample_pred(V_pred, V_tr, i, KSTEPS):
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

    kstep_V_pred_ls = []
    for j in range(KSTEPS):
        kstep_V_pred_ls.append(mvnormal.sample())  # cat [-1, num_person, 2]
    kstep_V_pred = torch.cat(kstep_V_pred_ls,dim=0) #[1*20, num_person, 2]
    
    V_this = V_tr.squeeze()[i:i+1,:,:] #[1, num_person, 2]

    distance = F.pairwise_distance(kstep_V_pred.reshape(-1,2), V_this.repeat(KSTEPS,1,1).reshape(-1,2), p=2).reshape(-1,kstep_V_pred.size()[1])
    index=torch.argmin(distance,dim=0)
    index=index.reshape(1,kstep_V_pred.size()[1],1).repeat(1,1,2)
    V_pred_result=torch.gather(kstep_V_pred, 0, index) #[-1, num_person, 2]

    return V_pred_result


def val(model, device, loader, KSTEPS=20):
    model.eval()
    loss_batch = 0
    batch_count = 0

    num_batch = len(loader)
    V_pred_rel_to_abs_ksteps_ls, V_y_rel_to_abs_ls, mask_ls = [None] * num_batch, [None] * num_batch, [None] * num_batch

    for step, batch in enumerate(loader):
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

        V_tr_tmp_start = V_obs_tmp[:,:,-1:,:]
        V_tr_tmp = V_tr_tmp_start # [1, 2, 1, num_person]


        for i in range(V_tr.shape[1]):
            V_pred = model(V_obs_tmp, A_obs_tmp, V_tr_tmp, -1, V_tr_tmp) #  [1, 5, 1, num_person]
            output=  sample_pred(V_pred, V_tr, i, KSTEPS) #  [-1, num_person, 2]
            output = output.permute(2,0,1).unsqueeze(0)
            V_tr_tmp = torch.cat((V_tr_tmp, output), 2) # torch.Size([1, 2, 26, 64])

        V_pred = V_pred.permute(0, 2, 3, 1) #  [1, 12, num_person, 5]]

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()  #  [12, num_person, 5]]

        loss_task = graph_loss(V_pred,V_tr)
        loss_batch += loss_task.item()

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy()) # [8, num_person, 2]
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :]) # [12, num_person, 2] speed??? torch.Size([25, 64, 2])
        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_tr_tmp.squeeze().permute(1,2,0)[1:,:,:].data.cpu().numpy(), V_x[-1, :, :]) # torch.Size([25, 64, 2])

        V_pred_rel_to_abs_ksteps_ls[step] = V_pred_rel_to_abs  # np.ndarray
        V_y_rel_to_abs_ls[step] = V_y_rel_to_abs  # np.ndarray

    
    loss=loss_batch/batch_count
    distance = F.pairwise_distance(torch.stack(V_pred_rel_to_abs_ksteps_ls), torch.stack(V_y_rel_to_abs_ls), p=3)
    final = torch.sum(torch.sum(distance,dim=0),dim=1)/(distance.size()[0]*distance.size()[2])

    return loss,final


def main():

    args = parse_args()

    output_dir = get_output_dir(args.folder_date, args.dataset, args.exp)
    # copy_source(args.run_file, output_dir)
    # copy_source(args.config_file, output_dir)
    
    # set_gpu(7)
    set_seed(args.seed)
    set_cuda(deterministic=args.gpu_deterministic)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger = setup_logging('job{}'.format(0), output_dir, console=True)
    logger.info(args)
    logger.info("Training initiating....")

    # Define the model
    model = SGTN(args).to(device)

    # Data loader
    loader_train, loader_val, loader_test = get_dataloader(args, logger)

    # Optimizer settings
    optimizer = NoamOpt(args.emb_size, args.factor, len(loader_train)*args.warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    # save argument once and for all
    with open(output_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    logger.info('Checkpoint dir:{:s}' .format(output_dir))

    metrics = {'train_loss': [], 'task_loss': [], 'contrast_loss': [], 'val_loss': []}


    # Start training
    
    disbig = 999.0
    best_epoch = 0

    for epoch in range(args.num_epochs):

        logger.info('Training ...')
        time_start = time.time()
        train_loss = train(model, optimizer, device, loader_train, args)
        time_elapsed = time.time() - time_start
        logger.info('TRAIN: Epoch:{:d}, train loss:{:.4f}'.format(epoch, train_loss))
        logger.info('Time to train once: {:.4f} s for dataset {:s}'.format(time_elapsed, args.dataset))
        
        logger.info("Testing ....")
        time_start = time.time()
        if (args.newTrans==1):
            loss, final = val_test(model, device, loader_test, args.KSTEPS)        
        else:
            loss, final = val(model, device, loader_test, args.KSTEPS)        
        time_elapsed = time.time() - time_start


        logger.info('TEST: Epoch:{:d}, val_test loss:{:.4f}'.format(epoch,loss))
        logger.info('Time to test once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))

        if (args.dataset=='car'):
            logger.info("dis1: {:.4f}, dis2: {:.4f}, dis3: {:.4f}, dis4: {:.4f}, dis5: {:.4F}".format(final[4],final[9],final[14],final[19], final[24]))
            if(final[24]<disbig):
                best_epoch = epoch
                disbig = final[24]
        else:
            logger.info("ade: {:.4f}, fde: {:.4f}".format(sum(final)/12,final[-1]))
            if(final[-1]<disbig):
                best_epoch = epoch
                disbig = final[-1]


        logger.info('Best epoch up to now is {}'.format(best_epoch))

        logger.info('*'*30)

        torch.save(model.state_dict(), output_dir + 'epoch{:03d}_val_best.pth'.format(epoch))
        shutil.copy(output_dir+'epoch{:03d}_val_best.pth'.format(best_epoch), output_dir + 'val_best.pth')

if __name__ == '__main__':
    main()
