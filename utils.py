import os
import sys
import math

import torch
import numpy as np
import networkx as nx
import logging

from torch.utils.data import Dataset
from tqdm import tqdm



def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len): 
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)):  
            V[s,h,:] = step_rel[h] 
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm 
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def interpolate_traj(traj, num_interp=4):
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1) 
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2):
    ph = predicted_traj.shape[0]
    num_interp = 4
    assert predicted_trajs_all.shape[0] > 1

    dense_all = interpolate_traj(predicted_trajs_all, num_interp) 
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp) 
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1) 
    mask = distances[:, 0] > 0  
    return (distances[mask].min(axis=0) < thres) 

def compute_col_pred(predicted_traj, predicted_trajs_all, mask_nei=[], thres=0.2):
    ph = predicted_traj.shape[0]
    num_interp = 4
    assert predicted_trajs_all.shape[0] > 1

    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)  
    distances = distances[mask_nei,:]

    mask = distances[:, 0] > 0 

    if (distances[mask].shape[0]==1):
        return (distances[mask][0] < thres)
    elif(distances[mask].shape[0]>1):
        return (distances[mask].min(axis=0) < thres)
    else:
        return np.repeat(False, ((ph-1)*5+1))

def adjConcat(a, b):

    for i in range(a.size()[0]):
        lena = len(a[i])
        lenb = len(b[i])
        left = np.row_stack((a[i], np.zeros((lenb, lena))))  
        right = np.row_stack((np.zeros((lena, lenb)), b[i]))  
        if (i == 0):
            result = np.expand_dims(np.hstack((left, right)),axis=0)  
        else:
            temp = np.expand_dims(np.hstack((left, right)),axis=0)
            result = torch.cat((torch.tensor(result),torch.tensor(temp)),0)
    return result


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'data.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, logger, 
        obs_len=8, pred_len=12, skip=20, threshold=0.002,
        min_ped=1, delim='tab',norm_lap_matr = True, 
        k=64, data_use='graph_data_64.dat'
        ):
        # 'tab'  'space' 
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr # True
        self.mini_bs = k

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        graph_data_path_use = os.path.join(self.data_dir, data_use)
        if not os.path.exists(graph_data_path_use):
            for path in all_files:
                if '.txt' not in path:
                    continue
                data = read_file(path, delim)
                logger.info("Processing Data ....."+str(path))
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len + 1) / skip))

                for person_idx in range(0, num_sequences * self.skip + 1, skip):
                    if(len(frame_data[person_idx:person_idx + self.seq_len])==0):
                        continue
                    curr_seq_data = np.concatenate(
                        frame_data[person_idx:person_idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                            self.seq_len))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                            self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    logger.info("Processing Data ....."+str(person_idx)+'/'+str(num_sequences * self.skip + 1)+'('+str(skip)+')')
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                    ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - person_idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - person_idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq = curr_ped_seq
                        if curr_ped_seq.shape[1] != self.seq_len:
                            continue
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered) 
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered]) 
                        seq_list.append(curr_seq[:num_peds_considered]) 
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered]) 
                    logger.info('seq_list: '+ str(len(seq_list)))

            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]

            # Convert to Graphs
            graph_data_path = os.path.join(self.data_dir, 'graph_data.dat')
            if not os.path.exists(graph_data_path):
                # process graph data from scratch
                self.v_obs = []
                self.A_obs = []
                self.v_pred = []
                self.A_pred = []

                logger.info("Processing Data .....")

                pbar = tqdm(total=len(self.seq_start_end))
                for ss in range(len(self.seq_start_end)):
                    pbar.update(1)

                    start, end = self.seq_start_end[ss]

                    v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
                    self.v_obs.append(v_.clone())
                    self.A_obs.append(a_.clone())
                    v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
                    self.v_pred.append(v_.clone())
                    self.A_pred.append(a_.clone())
                pbar.close()
                graph_data = {'v_obs': self.v_obs, 'A_obs': self.A_obs, 'v_pred': self.v_pred, 'A_pred': self.A_pred}
                torch.save(graph_data, graph_data_path)
            else:
                graph_data = torch.load(graph_data_path)
                self.v_obs, self.A_obs, self.v_pred, self.A_pred = graph_data['v_obs'], graph_data['A_obs'], graph_data['v_pred'], graph_data['A_pred']
                logger.info('Loaded pre-processed graph data at {:s}.'.format(graph_data_path))

            # prepare safe trajectory mask
            # if not os.path.exists(graph_data_path):
            self.safe_traj_masks = [] 
            for batch_idx in range(len(self.seq_start_end)):
                start, end = self.seq_start_end[batch_idx]
                pred_traj_gt = self.pred_traj[start:end, :]  

                num_person = pred_traj_gt.size(0)
                safety_gt = torch.zeros(num_person).bool()   
                label_tarj_all = pred_traj_gt.permute(0, 2, 1).cpu().numpy()  
                for person_idx in range(num_person):
                    label_traj_primary = label_tarj_all[person_idx]
                    cur_traj_col_free = np.logical_not(compute_col(label_traj_primary, label_tarj_all).max())
                    safety_gt[person_idx] = True if cur_traj_col_free else False
                self.safe_traj_masks.append(safety_gt)


        graph_data_path_use = os.path.join(self.data_dir, data_use)
        if not os.path.exists(graph_data_path_use):

            mini_bs=self.mini_bs
            flag = True #first item

            self.obs_traj_list = []
            self.pred_traj_list = []
            self.obs_traj_rel_list = []
            self.pred_traj_rel_list = []
            self.non_linear_ped_list = []
            self.loss_mask_list = []
            self.v_obs_list = []
            self.A_obs_list = []
            self.v_pred_list = []
            self.A_pred_list = []
            if 'train' in data_dir:
                self.safe_traj_masks_list = []

            for index in range(len(self.seq_start_end)):
                start, end = self.seq_start_end[index]

                if flag:
                    obs_traj_set = self.obs_traj[start:end, :].type(torch.FloatTensor)
                    pred_traj_set = self.pred_traj[start:end, :].type(torch.FloatTensor)
                    obs_traj_rel_set = self.obs_traj_rel[start:end, :].type(torch.FloatTensor)
                    pred_traj_rel_set = self.pred_traj_rel[start:end, :].type(torch.FloatTensor)
                    non_linear_ped_set = self.non_linear_ped[start:end].type(torch.FloatTensor) 
                    loss_mask_set = self.loss_mask[start:end, :].type(torch.FloatTensor)
                    v_obs_set = self.v_obs[index].type(torch.FloatTensor) 
                    A_obs_set = self.A_obs[index].type(torch.FloatTensor) 
                    v_pred_set = self.v_pred[index].type(torch.FloatTensor) 
                    A_pred_set = self.A_pred[index].type(torch.FloatTensor) 
                    if 'train' in data_dir:
                        safe_traj_masks_set =self.safe_traj_masks[index]
                    flag = False
                else: 
                    obs_traj_set = torch.cat((obs_traj_set, self.obs_traj[start:end, :]),0)
                    pred_traj_set = torch.cat((pred_traj_set,self.pred_traj[start:end, :]),0)
                    obs_traj_rel_set = torch.cat((obs_traj_rel_set, self.obs_traj_rel[start:end, :]),0)
                    pred_traj_rel_set = torch.cat((pred_traj_rel_set, self.pred_traj_rel[start:end, :]),0)
                    non_linear_ped_set = torch.cat((non_linear_ped_set, self.non_linear_ped[start:end]),0)
                    loss_mask_set = torch.cat((loss_mask_set, self.loss_mask[start:end, :]),0)
                    v_obs_set = torch.cat((v_obs_set, self.v_obs[index]),1)
                    A_obs_set = adjConcat(A_obs_set,  self.A_obs[index])
                    v_pred_set = torch.cat((v_pred_set,self.v_pred[index]),1)
                    A_pred_set = adjConcat(A_pred_set,  self.A_pred[index])
                    if 'train' in data_dir:
                        safe_traj_masks_set = torch.cat((safe_traj_masks_set, self.safe_traj_masks[index]),0)

                    
                if obs_traj_set.size()[0]>=mini_bs:

                    self.obs_traj_list.append(obs_traj_set[0:mini_bs,:,:])
                    self.pred_traj_list.append(pred_traj_set[0:mini_bs,:,:])
                    self.obs_traj_rel_list.append(obs_traj_rel_set[0:mini_bs,:,:])
                    self.pred_traj_rel_list.append(pred_traj_rel_set[0:mini_bs,:,:])
                    self.non_linear_ped_list.append(non_linear_ped_set[0:mini_bs])
                    self.loss_mask_list.append(loss_mask_set[0:mini_bs,:])
                    self.v_obs_list.append(v_obs_set[:,0:mini_bs,:])
                    self.A_obs_list.append(A_obs_set[:,0:mini_bs,0:mini_bs])
                    self.v_pred_list.append(v_pred_set[:,0:mini_bs,:])
                    self.A_pred_list.append(A_pred_set[:,0:mini_bs,0:mini_bs])
                    if 'train' in data_dir:
                        self.safe_traj_masks_list.append(safe_traj_masks_set[0:mini_bs])
                    flag = True

            if 'train' in data_dir:
                graph_data_use = {'obs_traj_list': self.obs_traj_list, 
                            'pred_traj_list': self.pred_traj_list, 
                            'obs_traj_rel_list': self.obs_traj_rel_list, 
                            'pred_traj_rel_list': self.pred_traj_rel_list,
                            'non_linear_ped_list': self.non_linear_ped_list, 
                            'loss_mask_list': self.loss_mask_list, 
                            'v_obs_list': self.v_obs_list, 
                            'A_obs_list': self.A_obs_list,
                            'v_pred_list': self.v_pred_list, 
                            'A_pred_list': self.A_pred_list, 
                            'safe_traj_masks_list': self.safe_traj_masks_list 
                            }
            else:
                graph_data_use = {'obs_traj_list': self.obs_traj_list, 
                            'pred_traj_list': self.pred_traj_list, 
                            'obs_traj_rel_list': self.obs_traj_rel_list, 
                            'pred_traj_rel_list': self.pred_traj_rel_list,
                            'non_linear_ped_list': self.non_linear_ped_list, 
                            'loss_mask_list': self.loss_mask_list, 
                            'v_obs_list': self.v_obs_list, 
                            'A_obs_list': self.A_obs_list,
                            'v_pred_list': self.v_pred_list, 
                            'A_pred_list': self.A_pred_list
                            }

            torch.save(graph_data_use, graph_data_path_use)
        else:
            graph_data_path_use = os.path.join(self.data_dir, data_use)
            graph_data_use = torch.load(graph_data_path_use)
            self.obs_traj_list = graph_data_use['obs_traj_list']
            self.pred_traj_list = graph_data_use['pred_traj_list']
            self.obs_traj_rel_list = graph_data_use['obs_traj_rel_list']
            self.pred_traj_rel_list = graph_data_use['pred_traj_rel_list']
            self.non_linear_ped_list = graph_data_use['non_linear_ped_list']
            self.loss_mask_list = graph_data_use['loss_mask_list']
            self.v_obs_list = graph_data_use['v_obs_list']
            self.A_obs_list = graph_data_use['A_obs_list']
            self.v_pred_list = graph_data_use['v_pred_list']
            self.A_pred_list = graph_data_use['A_pred_list']
            if 'train' in data_dir:
                self.safe_traj_masks_list= graph_data_use['safe_traj_masks_list']

    def __len__(self):
        return len(self.obs_traj_list)

    def __getitem__(self, index):
        if 'train' in self.data_dir:
            out = [
                self.obs_traj_list[index], self.pred_traj_list[index],
                self.obs_traj_rel_list[index], self.pred_traj_rel_list[index],
                self.non_linear_ped_list[index], self.loss_mask_list[index],
                self.v_obs_list[index], self.A_obs_list[index],
                self.v_pred_list[index], self.A_pred_list[index], 
                self.safe_traj_masks_list[index]
            ]
        else:
            out = [
                self.obs_traj_list[index], self.pred_traj_list[index],
                self.obs_traj_rel_list[index], self.pred_traj_rel_list[index],
                self.non_linear_ped_list[index], self.loss_mask_list[index],
                self.v_obs_list[index], self.A_obs_list[index],
                self.v_pred_list[index], self.A_pred_list[index]
            ]
        return out

def main():

    # dataName =['eth','hotel','univ','zara1','zara2']
    dataName =['car']
    setName=['test','val','train']

    checkpoint_dir = '../../../scratch/data/SGTN/datasets/' + dataName[0]+ '/'
    logger = setup_logging('job{}'.format(0), checkpoint_dir, console=True)

    for j in range(3):
        data_dir='../../../scratch/data/SGTN/datasets/'+dataName[0]+'/'+setName[j]+'/'
        TrajectoryDataset(
            data_dir=data_dir,
            logger=logger,
            obs_len=15,
            pred_len=25,
            skip=20,
            threshold=0.002,
            min_ped=1,
            delim='space',  # 'space'
            norm_lap_matr = True,
            k=128,
            data_use='graph_data_128.dat')

    

if __name__ == '__main__':
    main()