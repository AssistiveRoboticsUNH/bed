import numpy as np
from collections import OrderedDict
 
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
 
import robomimic.models.policy_nets as PolicyNets 
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from robomimic.utils.dataset import SequenceDataset

from tqdm import tqdm 
import time 
 
from copy import deepcopy
from contextlib import contextmanager
import json
from tqdm import tqdm
from scipy import interpolate 
import torch.utils.data

'''
both for bc and bc_wr
'''

def backprop_for_loss(net, optim, loss, max_grad_norm=None, retain_graph=False):

    # backprop
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clipping
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    #clip w
    if 'w' in net.nets.keys():
        with torch.no_grad():
            net.nets["w"].weight.clamp_(min=0.0, max=1.0)

    # compute grad norms
    grad_norms = 0.
    for p in net.parameters():
        # only clip gradients for parameters for which requires_grad is True
        if p.grad is not None:
            grad_norms += p.grad.data.norm(2).pow(2).item()

    # step
    optim.step()

    return grad_norms

def load_config(config_filepath):
    ext_cfg = json.load(open(config_filepath, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
 
    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.lock()
    return config, device

def extract_last_obs(demo):
    """ 
    return last step as a batch
    """
    obss=demo['obs']
    last_obs=OrderedDict()
    for key in obss.keys():
        last_obs[key]=obss[key][-1]
        last_obs[key]=last_obs[key][None,:]

    batch=OrderedDict()
    batch['obs']=last_obs
    for key in batch['obs'].keys():
        batch['obs'][key] = torch.tensor(batch['obs'][key][:,None])

    batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
    batch['goal_obs'] = None
    return batch

def estimate_first_goal(trainset, model, device):
    """ 
    Taking last obs from all the demos estimate latent goal as expected goal.
    Use weights as the probability.
    """ 
    encoder=model.nets["policy"].nets["encoder"] 
    encodeds=[]
    for i in range(len(trainset.demos)):
        demo=trainset.get_trajectory_at_index(i) 
        batch=extract_last_obs(demo)
        input_batch = TensorUtils.to_device(TensorUtils.to_float(batch), device)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

        encoded=encoder(**input_batch)
        encodeds.append(encoded.detach().cpu().numpy())

    encodeds=np.concatenate(encodeds, axis=0)

    w=model.nets["policy"].nets["w"].weight.detach().cpu().numpy()
    w=w/np.sum(w)

    first_goal=w @ encodeds
    first_goal = torch.tensor(first_goal).to(device).float()
    return first_goal

def train_bed_demos_batch(model, trainset, epoch, M, m_start, first_goal, device, batch_d=12 , wscale=10, gscale = 5):
    """ 
    train the model on a batch of demos
    M = m*nump_demo
    m_start = start adding consitency loss after m_start epochs
    first_goal = the first goal estimated using the initial weights
    """ 
    
    #divide the indices into batches
    num_steps=len(trainset.demos)
    indices=list(range(num_steps)) 
    k=num_steps//batch_d 
    if k*batch_d< num_steps: k=k+1 
    batch_is=[indices[i*batch_d : i*batch_d +batch_d] for i in range(k)]

    # train the model on each batch
    obs_normalization_stats=None  
    total_loss=0
    total_goal_loss=0
    all_latents=[]
    for b in tqdm(range(k)):
        batch_i=batch_is[b]

        all_losses=[]
        last_latents = []
        obs_normalization_stats=None 
        for i in batch_i:
            demo=trainset.get_trajectory_at_index(i)
            demo['actions'] = demo['actions'][:,None]
            # demo_goal=demo['obs']['robot0_eef_pos'][-1]
            # goal_loss=np.linalg.norm(demo_goal - goal)
            # g_lossess.append(goal_loss)
            for key in demo['obs'].keys():
                demo['obs'][key] = demo['obs'][key][:,None]
                
            demo['actions'] =TensorUtils.to_tensor(demo['actions'])
            for key in demo['obs'].keys():
                demo['obs'][key]=TensorUtils.to_tensor(demo['obs'][key])
            batch=demo 
            input_batch = model.process_batch_for_training(batch)
            input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)

            # forward and backward pass
            # info = model.train_on_batch(input_batch, epoch, validate=validate)
            info=OrderedDict() 
            predictions = model._forward_training(input_batch)
            losses = model._compute_losses(predictions, input_batch)
            all_losses.append(losses['action_loss'])

            last_latent = model.nets["policy"].last_latent
            last_latents.append(last_latent)
            all_latents.append(last_latent.detach().cpu().numpy())

        all_losses=torch.stack(all_losses)
        last_latents = torch.stack(last_latents).to(device).float()
        
        weighted_g_loss= torch.tensor([0])
         
        if epoch > m_start:# config.train.mstartat:
            weighted_loss=(model.nets["policy"].nets["w"].weight[:, batch_i]*2*wscale) @ all_losses.float()  
            m_loss=(M-torch.sum(model.nets["policy"].nets["w"].weight))**2 
  
            gd = torch.norm(first_goal - last_latents, dim=1)
            weighted_g_loss=(model.nets["policy"].nets["w"].weight[:, batch_i]*2*gscale) @ gd.to(device).float()  

            final_loss=weighted_loss + m_loss + weighted_g_loss
            final_loss=final_loss.float()
 

        else:
            final_loss=torch.sum(all_losses)

        policy_grad_norms = backprop_for_loss(
            net=model.nets["policy"],
            optim=model.optimizers["policy"],
            loss=final_loss,
        )
        total_loss+=final_loss.item()
        total_goal_loss+=weighted_g_loss.item()

    #One step training is done, now update the latent goal estimate 
    all_latents2 =np.vstack(all_latents)
    w=model.nets["policy"].nets["w"].weight.detach().cpu().numpy()
    w=w/np.sum(w) 
    first_goal=w @ all_latents2
    first_goal = torch.tensor(first_goal).to(device).float()
     
     
    info=OrderedDict()
    info['loss']=total_loss
    info['goal_loss']=total_goal_loss
    return info, first_goal

def train_bed_path_demos_batch(model, trainset, epoch, M, m_start, first_goal, device, maxlen, batch_d=12 , wscale=10, gscale = 5, k=1):
    """ 
    train the model on a batch of demos
    M = m*nump_demo
    m_start = start adding consitency loss after m_start epochs
    first_goal = the first goal estimated using the initial weights
    """ 
    
    #divide the indices into batches
    num_steps=len(trainset.demos)
    indices=list(range(num_steps)) 
    k=num_steps//batch_d 
    if k*batch_d< num_steps: k=k+1 
    batch_is=[indices[i*batch_d : i*batch_d +batch_d] for i in range(k)]

    # train the model on each batch
    obs_normalization_stats=None  
    total_loss=0
    total_goal_loss=0
    all_latents=[]
    for b in tqdm(range(k)):
        batch_i=batch_is[b]

        all_losses=[]
        last_latents = []
        obs_normalization_stats=None 
        for i in batch_i:
            demo=trainset.get_trajectory_at_index(i)
            demo['actions'] = demo['actions'][:,None]
            # demo_goal=demo['obs']['robot0_eef_pos'][-1]
            # goal_loss=np.linalg.norm(demo_goal - goal)
            # g_lossess.append(goal_loss)
            for key in demo['obs'].keys():
                demo['obs'][key] = demo['obs'][key][:,None]
                
            demo['actions'] =TensorUtils.to_tensor(demo['actions'])
            for key in demo['obs'].keys():
                demo['obs'][key]=TensorUtils.to_tensor(demo['obs'][key])
            batch=demo 
            input_batch = model.process_batch_for_training(batch)
            input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)

            # forward and backward pass
            # info = model.train_on_batch(input_batch, epoch, validate=validate)
            info=OrderedDict() 
            predictions = model._forward_training(input_batch)
            losses = model._compute_losses(predictions, input_batch)
            all_losses.append(losses['action_loss'])

            # last_latent = model.nets["policy"].last_latent  
            last_latent = model.nets["policy"].latent
            last_latents.append(last_latent)
            all_latents.append(last_latent.detach().cpu().numpy())


        all_losses=torch.stack(all_losses)
        # last_latents = torch.stack(last_latents).to(device).float()
        
        weighted_g_loss= torch.tensor([0])
         
        if epoch > m_start:# config.train.mstartat:
            weighted_loss=(model.nets["policy"].nets["w"].weight[:, batch_i]*2*wscale) @ all_losses.float()  
            m_loss=(M-torch.sum(model.nets["policy"].nets["w"].weight))**2 
  
            # gd = torch.norm(first_goal - last_latents, dim=1)
            # weighted_g_loss=(model.nets["policy"].nets["w"].weight[:, batch_i]*2*gscale) @ gd.to(device).float()  

            encodeds=last_latents
            interpolated=[]
            for i in range(len(encodeds)):
                matrix=encodeds[i].detach().cpu().numpy()
                original_shape = matrix.shape
                new_shape = (maxlen, original_shape[1])

                interpolator = interpolate.interp1d(np.arange(original_shape[0]), matrix, kind='linear', axis=0)
                new_matrix = interpolator(np.linspace(0, original_shape[0]-1, new_shape[0]))
                interpolated.append(new_matrix)

            interps=np.stack(interpolated, axis=0)

            df=first_goal.detach().cpu().numpy() - interps
            norms = np.linalg.norm(df, axis=(1, 2))            #TODO: changed to Mahalanobis distance
            norms=torch.tensor(norms)
            weighted_g_loss=(model.nets["policy"].nets["w"].weight[:, batch_i]*2*gscale) @ norms.to(device).float() 

            final_loss=weighted_loss + k*m_loss + weighted_g_loss
            final_loss=final_loss.float()
 

        else:
            final_loss=torch.sum(all_losses)

        policy_grad_norms = backprop_for_loss(
            net=model.nets["policy"],
            optim=model.optimizers["policy"],
            loss=final_loss,
        )
        total_loss+=final_loss.item()
        total_goal_loss+=weighted_g_loss.item()

    #One step training is done, now update the latent goal estimate 
    encodeds=all_latents
    interpolated=[]
    for i in range(len(encodeds)):
        matrix=encodeds[i]
        original_shape = matrix.shape
        new_shape = (maxlen, original_shape[1])

        interpolator = interpolate.interp1d(np.arange(original_shape[0]), matrix, kind='linear', axis=0)
        new_matrix = interpolator(np.linspace(0, original_shape[0]-1, new_shape[0]))
        interpolated.append(new_matrix)

    interps=np.stack(interpolated, axis=0)

    w=model.nets["policy"].nets["w"].weight.detach().cpu().numpy()
    w=w/np.sum(w)
    w=w.reshape(-1,1)
    path_estimation =np.tensordot(w, interps, axes=([0], [0]))
    path_estimation=path_estimation.squeeze()
    first_goal = torch.tensor(path_estimation).to(device).float()
     
    info=OrderedDict()
    info['loss']=total_loss
    info['goal_loss']=total_goal_loss
    return info, first_goal

