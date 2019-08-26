# SpherePHD
Reproduce SpherePHD with python codes (https://arxiv.org/pdf/1811.08196.pdf)

# Usage
- Using `DataLoader.py` to make `data.npy` and `label.npy`
  The function will sample the Sphere pixel automatically
- Get the reconstruction information from `makedata.py` to get division log.
- run `train.py` to train your own data, which includes simpleCNN and autoencoder

# Implement Detail
- Sampling the pixel from panorama should be different from the original paper.
  I use direct projection when doing subdivsions.
- The results on Stanford2D3D and spherical MNIST is close to the original paper

# Reference
example code from authors
https://github.com/KAIST-vilab/SpherePHD_public
