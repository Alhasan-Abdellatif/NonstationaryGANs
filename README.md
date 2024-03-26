# Generation of Nonstationary geological fields using GANs
Pytorch implementation of the method used in "Generation of non-stationary stochastic fields using Generative Adversarial Networks" https://arxiv.org/abs/2205.05469. 

The work is an extenstion of the previous work https://github.com/Alhasan-Abdellatif/cGANs where spatial conditioning is implemented using the SPADE algorithm https://github.com/NVlabs/SPADE to generate geological samples conditioned on soft data.


## Requirements

* Python 3.8.10
* PyTorch 1.12.1
* NumPy
* matplotlib
* torchvision

## Training


To run conditional GAN using SPADE algorithm on images in ```datasets/images/ ``` and their labels (soft probabilities e.g., 4x4) save models in ```results``` :
```
python train.py --data_path datasets/images  --labels_path datasets/labels/ --data_ext txt  --img_ch 1  --zdim 128 --spec_norm_D --x_fake_GD --y_real_GD --n_cl 1 --cgan --G_cond_method conv1x1 --D_cond_method conv1x1 --batch_size 32  --epochs  100 --smooth --save_rate 10  --ema --dev_num 1  --att  --fname results
```

The sample notebook provides some examples on how to use the trained models to genereate different geological facies controlled spatailly by a soft map input.  
