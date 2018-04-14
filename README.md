Python code for  "Accelerated Inference for Latent Variable Models" (Zhang and Perez-Cruz, 2017) as well as code for distributed slice sampler (Ge et al., 2015) and Algorithm 8 (Neal, 2000). Parallelization is carried out through MPI (using mpi4py).

Run slice sampler through "slice_sampler_DP_mult.py". Run Algorithm 8 by setting "collapsed = True" and "bin_threshold" to be strictly greater than the number of iterations in "parallel_acceleratedDP_mult_v2.py".

To generate big MNIST example, install "infimnist" (from http://leon.bottou.org/projects/infimnist) and run the following commands in the "data" directory

infimnist pat 1 1000000 > big_mnist_train

infimnist pat 1000001 1010000 > big_mnist_test

To generate grayscale CIFAR-10 data, run data/cifar10_grayscale.py. Extended Yale face dataset available at http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
