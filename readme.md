### Behavior cloning for Error Discovery
This repository contains the code for the paper "Self Supervised Detection of Incorrect Human Demonstrations: A Path
Toward Safe Imitation Learning by Robots in the Wild" by Noushad Sojib & Momotaz Begum.

BED is a BC model with an additional parameter vector $w$ of length $|D|$. It utilize a loss function that penalize different kinds of inconsistency and help to learn $w_i\approx1$ for good demos and $w_i\approx0$ for bad demos. As bad demos add more loss the the total loss, discarding them (by assigning $w_i=0$) will reduce the total loss and help to detect the bad demos.

### Installation
* Install Robomimic and Robosuite as follows https://robomimic.github.io/docs/introduction/installation.html

* Add the two lines in the 'forward' function of class MIMO_MLP in ["robomimic/robomimic/models/obs_nets.py line 585"](https://github.com/ARISE-Initiative/robomimic/blob/9273f9cce85809b4f49cb02c6b4d4eeb2fe95abb/robomimic/models/obs_nets.py#L585) as shown below.

```python
    def forward(self, **inputs):
        enc_outputs = self.nets["encoder"](**inputs) 
        self.last_latent = enc_outputs[-1,:]    #add this line
        self.latent=enc_outputs                 #add this line
        mlp_out = self.nets["mlp"](enc_outputs)
        return self.nets["decoder"](mlp_out)
``` 

### Download the data 
Use this [link](https://universitysystemnh-my.sharepoint.com/:f:/g/personal/mb1215_usnh_edu/EpIt98g81rBBpVtUxi8pldsB-FthJ8I5FA650TxfQS2Ydw?e=ja4JNI) to download the dataset. Put them in the bed/dataset folder. Or use the following commands
```bash
cd bed
mkdir dataset
cd dataset

# download can_task data
wget https://universitysystemnh-my.sharepoint.com/:u:/g/personal/mb1215_usnh_edu/EdaW2mZ4mRpGg0CKbTEwG5UBbKCxqXqlGnyIHdhL-o8Ahw?download=1 -O layman_v1_can_510.hdf5

# download square_task data
wget https://universitysystemnh-my.sharepoint.com/:u:/g/personal/mb1215_usnh_edu/ERbUWCBrp1xAj49yUOmoHJ8B4x6G_1EgNaUNHiZsSd_V7g?download=1 -O layman_v1_square_180.hdf5

# download lift_task data
wget https://universitysystemnh-my.sharepoint.com/:u:/g/personal/mb1215_usnh_edu/EQyR2TBr5aZKusxWCnn0Y6ABJJXDNeHZL2vhUCq-4__9Sw?download=1 -O layman_v1_lift_260.hdf5

```

### create configuration file
We use the same configuration file as Robomimic. Please see the "bed/configs/can/bed_layman_can_p20b.json" file for an example configuration file. You can create a similar configuration file for other tasks. Based on the dataset you may want to change the following three lines.
```bash
    "data": "dataset/layman_v1_can_510.hdf5",
    "output_dir": "/home/ns1254/bed/training_data",
    "hdf5_filter_key": "p20b",
```


### Training BED 

Run the following command to train the BED model
```bash
python bed_training_path.py --config config_full_path.json --m 0.8 --accelerate 40 --gscale 5
```
Example: Train BED on can data to detect 80% as good and 20% as bad. You can press Ctrl+C for early stopping.
```bash
python bed_training_path.py --config /home/ns1254/bed/configs/can/bed_layman_can_p20b.json --m 0.8 --accelerate 40 --gscale 5
```
Explnation of the arguments:
* --config: path to the configuration file
* --m: percentage of demos we want to keep
* --accelerate: use higher learning rate after this epoch for faster convergence
* --gscale: importance of path loss

Expected <b>w</b>: As there are 150 demos total in the can dataset, for m=0.8 we expect 30 (150*0.2=30) of them will get $w\approx0$ and 120 of them will get $w\approx1$. Rounding will make them binary. Here is the expected w vector before rounding:
```bash
w:  [ 1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.
  1.    1.    1.    1.    1.    1.    1.    1.    0.49  1.    1.    1.
  1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.
  1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.
  1.    1.    1.    1.    1.    1.    1.    1.    1.    0.96  1.    1.
  1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.    1.
  0.73  1.    1.    1.    1.    0.99  1.    1.    1.    1.    1.    1.
  1.    1.    0.99  1.    1.    0.99  0.91  1.    1.    1.    1.    1.
  1.    0.62  0.99  0.99  1.    1.    1.    0.48  0.08  0.72  1.    1.
  0.84  1.    1.    1.    1.    1.    0.99  0.99  1.    1.    1.    1.
 -0.   -0.   -0.    0.51 -0.   -0.   -0.   -0.   -0.   -0.   -0.   -0.
 -0.   -0.   -0.   -0.   -0.   -0.   -0.    0.   -0.    0.59 -0.    0.
  0.37 -0.   -0.   -0.    0.    1.  ]
```

View the training log located in <a href="logs/20240408221524/logs">logs</a> to see how training looks like.
It tooks 47 minutes to train on a Single NVIDIA A40 GPU.



### Acknowledgement
* Robomimic: https://robomimic.github.io/
