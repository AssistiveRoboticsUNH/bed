import argparse
import json
import numpy as np
import time
import os
import psutil
import sys

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger
 
import torch
import torch.utils.data 

from bed_model import BED
from bed_utils import estimate_first_goal, train_bed_path_demos_batch




def main(args):

    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.lock()


    config.unlock()  
    config.train.mstartat=args.mstartat
    config.train.wscale =args.wscale
    config.train.m=args.m
    config.lock()

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    ) 


    # load dataset and initialize observation utilities
    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    #env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    ) 
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset) 

    demo_names = trainset.demos
    #save the demo names in the log directory
    with open(os.path.join(log_dir, 'demos.txt'), 'w') as f:
        for item in demo_names:
            f.write("%s\n" % item)


    #setting new values for config
    config.unlock()
    config.algo.num_demo = len(trainset.demos) #TODO: remove this line safely.
    config.train.num_demo = len(trainset.demos)
    config.train.M = int( config.train.m*len(trainset.demos) )
    config.train.num_bad = config.train.num_demo - config.train.M
    config.lock()
    
    
    algo_kwargs={}

    model=BED(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=7,
        device=device,
        **algo_kwargs
    )


    #if goal_mode not none, then estimate first latent goal
    first_goal=estimate_first_goal(trainset, model, device)
    

    np.set_printoptions(precision=2, suppress=True)
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)


    lengths=[ trainset.get_trajectory_at_index(i)['actions'].shape[0] for i in range(len(trainset.demos))]
    maxlen=np.max(lengths)

    try:
        # main training loop  
        start_time = time.time()
        for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
            #for faster convergence.
            if epoch==args.accelerate:
                model.optimizers["policy"].param_groups[0]['lr']=0.01

            step_log, first_goal =train_bed_path_demos_batch(model, trainset, epoch, M=config.train.M, m_start=config.train.mstartat, first_goal=first_goal, device=device, maxlen=maxlen, batch_d=args.batch_d , wscale=args.wscale, gscale=args.gscale, k=args.k)
            model.on_epoch_end(epoch) 

            # after each epoch, log all the metrics
            ws=model.nets["policy"].nets["w"].weight.detach().cpu().numpy()
            n50s = (ws<0.5).sum()          #how many weights are less than 0.5
            step_log['sum(w)']=float(ws.sum())
            step_log['n50s']=int(n50s) 
            step_log["dt"] =  time.time() - start_time 
            step_log["dt"] = step_log["dt"]/60

            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            print("w: ", ws.ravel())
        
            
            # Finally, log memory usage in MB
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))
        
            #stop if (N-M) demos get 0 weights.
            if n50s>=config.train.num_bad:
                print("Early stopping")
                break

        # terminate logging
        data_logger.close()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        print("interrupted at epoch ", epoch)


    #save the model
    epoch_ckpt_name = "epoch_{}".format(epoch)
    TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=None,
            shape_meta=shape_meta,
            ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
            obs_normalization_stats=None,
    )


    #lets print w and bmask
    w=ws 
    bmask= np.round(w ).astype(np.int8)   

    bmask=bmask[0]
    print('-------------------')
    print(bmask.sum())
    print(bmask)
    print('-------------------')
    
    masked_0=[]
    for i, demo_name in enumerate(trainset.demos):
        if bmask[i]==0:
            masked_0.append(demo_name) 

    print(masked_0)
    print('-------------------')

    #save the weights and bmask as csv files
    #save w as csv
    np.savetxt(os.path.join(log_dir, 'w.csv'), w, fmt='%f', delimiter=',')
    np.savetxt(os.path.join(log_dir, 'bmask.csv'), bmask, fmt='%d', delimiter=',')
    with open(os.path.join(log_dir, 'masked_0.csv'), 'w') as f:
        for item in masked_0:
            f.write("%s\n" % item)

    
    np.savetxt(os.path.join(log_dir, 'w.txt'), w, fmt='%f')
    np.savetxt(os.path.join(log_dir, 'bmask.txt'), bmask, fmt='%d')
    with open(os.path.join(log_dir, 'masked_0.txt'), 'w') as f:
        for item in masked_0:
            f.write("%s\n" % item)
 

    print("find logs at: ", log_dir)
    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=0.8,
        help="percent of demos we want to keep as good demos",
    )
    parser.add_argument(
        "--mstartat",
        type=int,
        default=1,
        help="kick start optimizing w after this epoch",
    )
    parser.add_argument(
        "--wscale",
        type=int,
        default=20,
        help="importance of action loss in the total loss",
    )
    parser.add_argument(
        "--gscale",
        type=int,
        default=10,
        help="importance of goal/path loss in the total loss",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k",
    )

    parser.add_argument(
        "--accelerate",
        type=int,
        default=200,
        help="increase learning rate after this epoch",
    )
    parser.add_argument(
        "--batch_d",
        type=int,
        default=6,
        help="number of demos as batch, depends on GPU memory",
    )

    args = parser.parse_args()
    main(args)


# /home/ns1254/robomimic/robomimic/models/obs_nets.py
#585 self.latent=enc_outputs                 #TODO: remove?

# python bed_training_path.py --config /home/ubuntu/BED/franka_config.json --m 0.67 --accelerate 40 --gscale 5 --batch_d 1
