# sae-pytorch

Original : [[MATLAB version]](https://github.com/Elyorcv/SAE)

PyTorch implementation of Semantic AutoEncoder (SAE).

## How to Run
1. git clone https://github.com/hoseong-kim/sae-pytorch.git
2. Download 'awa_demo_data.mat'      
      * https://drive.google.com/file/d/1l0UVhhIU-SmtJ9hqk7OVOG9zNga9qt_I/view
3. python sae.py

## An Implementation of SAE in PyTorch
1. Set CUB, AwA, aP&Y, SUN, and ImageNet datasets.
    * Partially done (only for AwA dataset). 
    * Other datasets will also be available soon.
2. Extract deep features from various deep models, e.g., AlexNet, VGG16, VGG19, GoogleNet, Inception_v3, ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152.
    * Done, but tuning my source code to achieve results in this paper.
    * The source code will be available after reproducing.
3. Train a Semantic AutoEncoder (SAE).
    * Done.
4. Test unseen class data.
    * Done.

## Release Note

#### v1.0

* Bug fix

## Download Paper
Semantic Autoencoder for Zero-shot Learning: [[Paper Link (arXiv)]](https://arxiv.org/abs/1704.08345)
