# CompositeGAN_WhistleAugment

This is the repository of our paper "Learning Stage-wise GANs
 for Whistle Extraction in Time-Frequency Spectrograms" 
 published on IEEE Transactions on Multimedia (2023/3/1). 
 
## DeepWhistle model
The codes for generating training data and whistle 
extraction model training and evaluation are provided 
in our previous repository of [DeepWhistle](https://github.com/Paul-LiPu/DeepWhistle).
 
## Stage-wise GAN
#### WGANs for generating background noises and whistle contours
To train the WGAN model
> ```bash
> python train_batch.py --mode [pos | neg] --data_meta_dir [path_to_folder_containing_training_h5_files]
> ```
For the mode option, please choose 'pos' for generating whistle 
contours and 'neg' for generating background noise. 

The code is based on [https://github.com/caogang/wgan-gp](https://github.com/caogang/wgan-gp)

#### CycleGAN for generating whistles on background noise

The code is based on [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
