from collections import OrderedDict

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
 
import robomimic.models.policy_nets as PolicyNets 
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils 
from robomimic.algo import PolicyAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data


class BED(PolicyAlgo):
    """
    BC+w term
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        
        num_demo=self.algo_config.num_demo  #need to set this in config
        self.nets["policy"].nets["w"]=nn.Linear(num_demo,1,bias=False)
        
        #give all 50% probability.
        with torch.no_grad(): 
            self.nets["policy"].nets["w"].weight= nn.Parameter(torch.ones_like(self.nets["policy"].nets["w"].weight)*0.5)
    
        
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            info = OrderedDict()
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)  #backprop
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions
    
    def _compute_losses(self, predictions, batch):
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses
    
    def _train_step(self, losses):
        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        # log = super(BC_WR, self).log_info(info)
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if 'M' in info['losses']:
            log['M']=info['losses']['M'].item()
        if '01' in info['losses']:
            log['01']=info['losses']['01'].item()
            
        return log

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)

