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
*  can_task data <a href="https://universitysystemnh-my.sharepoint.com/:u:/g/personal/mb1215_usnh_edu/EdaW2mZ4mRpGg0CKbTEwG5UBbKCxqXqlGnyIHdhL-o8Ahw?e=gPEKTa">download</a>
* Layman V1.0 Full dataset [download link](https://universitysystemnh-my.sharepoint.com/:f:/g/personal/mb1215_usnh_edu/EpIt98g81rBBpVtUxi8pldsB-FthJ8I5FA650TxfQS2Ydw?e=ja4JNI)

### Training BED
Run the following command to train the BED model
```bash
python bed_training_path.py --config path/config.json --m 0.8 --accelerate 40 --gscale 5
```
Example: Train BED on can 80% data. You can press Ctrl+C for early stopping.
```bash
python bed_training_path.py --config path/configs/can/bed_can_510_p20b.json --m 0.8 --accelerate 40 --gscale 5
```
Explnation of the arguments:
* --config: path to the configuration file
* --m: percentage of demos we want to keep
* --accelerate: use higher learning rate after this epoch for faster convergence
* --gscale: importance of path loss

Expected <b>w</b>: As there are 150 demos total, we expect 30 of them will get $w\approx0$ and 120 of them will get $w\approx1$. Rounding will make them binary. Here is the expected w vector before rounding:
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
