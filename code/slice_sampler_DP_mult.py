# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 13:18:18 2018

Parallel Slice Sampler DP Mutltinomial Mixture Model

@author: Michael Zhang
"""
import argparse
import numpy as np
import time
import datetime
import os
from mpi4py import MPI
from sklearn.cluster import KMeans
from scipy.misc import logsumexp
from scipy.special import gammaln
from scipy.io import loadmat
from scipy.io import savemat
import mmap
import struct
from array import array as pyarray
from scipy.sparse import lil_matrix
from sys import exit

class SliceDP(object):

    def __init__(self, data, X_star, init_K=5, iters=1, alpha=1.,
                 prior_gamma=1., L=10, rand_init=True,fname="../figs/output",
                 max_time=86400.):
        """
        data: numpy array or string, training data. If numpy array then data is
              NxD file. If string, then the string is the filename of MNIST
              data generated from infimnist.
        init_K: int, number of initial clusters
        iters: int, number of MCMC iterations
        alpha: float, DP concentration parameter
        prior_gamma: float, parameter for Dirichlet prior on multinomial
                     likelihood
        L: int, number of iterations to run before triggering synchronization
           step
        X_star: numpy array or string, If numpy array then data is
                N_star x D file. If string, then the string is the filename of
                MNIST data generated from infimnist.
        rand_init: bool, if True then clusters initialized randomly.
        fname: string, directory of where to save output files
        max_time: float, maximum number of time (in seconds) to run sampler,
                  accelerated  stage is automatically stopped after max_time/2
                  seconds.
        """
        self.max_time = float(max_time)
        self.iters = int(iters)
        self.K = int(init_K)
        self.fname = str(fname)
        self.total_time = time.time()

        assert(self.K > 0)
        self.L = L
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.rand_init = bool(rand_init)
        self.alpha = alpha # parameter for dirichlet mixture
        assert(self.alpha > 0.)
#        self.prior_gamma = prior_gamma # parameter for dirichlet prior on multinomial
#        assert(self.prior_gamma > 0.)
        self.k_means = None

        if data=="../data/big_mnist_train":
            fname_img = os.path.abspath(data)
            with open(fname_img, 'r+') as fimg:
                m = mmap.mmap(fimg.fileno(), 0)
                magic_nr, self.N, rows, cols = struct.unpack(">IIII", fimg.read(16))

            if self.rank == 0:
                mnist_idx=np.array_split(xrange(self.N),self.P)
            else:
                mnist_idx=None
            mnist_idx = self.comm.scatter(mnist_idx)
            self.N_p = int(mnist_idx.size)
            self.D = int(rows*cols)
            self.X_local = lil_matrix(np.zeros((self.N_p, self.D), dtype=np.uint8))
            for idx, i in enumerate(mnist_idx):
                self.X_local[idx] =  pyarray("B",m[16+(i*self.D) : 16+((i+1)*self.D)])
            fimg.close()
            m.close()
        else:
            if self.rank==0:
                self.X = data.astype(int)
                self.N, self.D = self.X.shape
                self.X = np.array_split(self.X, self.P)
            else:
                self.N = None
                self.D = None
                self.X = None
            self.N = self.comm.bcast(self.N)
            self.D = self.comm.bcast(self.D)
            self.X_local = self.comm.scatter(self.X)
            self.X_local = self.X_local.reshape(-1,self.D).astype(int)
            self.N_p, _ = self.X_local.shape


        assert(prior_gamma > 0)
        self.prior_gamma = prior_gamma*np.ones(self.D)
        self.pi = np.random.dirichlet([self.alpha]*self.K)

        self.X = None
        self.i_star = 0
        self.P_star = 0
        self.slice_star = .01
        self.Z_init()

        if self.rank == 0:
            sync_iters = [it for it in xrange(self.iters) if (it % self.L == 0) or it==max(xrange(self.iters))]
            self.trace_size = len(sync_iters)
            self.L_dict = {it:L for L,it in enumerate(sync_iters)}
            self.K_trace= np.empty(self.trace_size)
            self.likelihood_trace = np.empty((self.trace_size,2))
            self.phi_pi = np.random.dirichlet(self.prior_gamma, size=self.K)

            if X_star=="../data/big_mnist_test":
                fname_img = os.path.abspath(X_star)
                with open(fname_img, 'r+') as fimg:
                    m = mmap.mmap(fimg.fileno(), 0)
                    magic_nr, self.N_star, rows, cols = struct.unpack(">IIII", fimg.read(16))

                if self.rank == 0:
                    mnist_idx=np.array_split(xrange(self.N_star),self.P)
                else:
                    mnist_idx=None
                mnist_idx = self.comm.scatter(mnist_idx)
                self.X_star = lil_matrix(np.zeros((mnist_idx.size, self.D), dtype=np.uint8))
                for idx, i in enumerate(mnist_idx):
                    self.X_local[idx] =  pyarray("B",m[(i*self.D)+16 : 16+((i+1)*self.D)])
                fimg.close()
                m.close()
                del fimg
                del m

            else:
                self.X_star = X_star
                if self.X_star is None:
                    self.N_star = None
                    self.predictive_likelihood = None
                else:
                    self.N_star, _ = self.X_star.shape
                    self.predictive_likelihood = np.zeros(self.trace_size)
        else:
            self.trace_size = None
            self.L_dict = None
            self.K_trace = None
            self.likelihood_trace = None
            self.X_star = None
            self.N_star = None
            self.predictive_likelihood = None
            self.phi_pi = None


        self.phi_pi = self.comm.bcast(self.phi_pi)
        self.slice_local = np.random.uniform(0, [self.pi[z_i] for z_i in self.Z_local])

        self.posterior_update3(it=0)



    def Z_init(self):
        if self.K > 1:
            if self.rand_init:
                self.k_means = None
                self.Z_local = np.random.choice(self.K,size=self.N_p) #np.zeros(self.N_p).astype(int)
            else:
                if self.rank == 0:
                    self.k_means = KMeans(n_clusters=self.K)
                    self.k_means.fit(self.X_local.astype(float))
                else:
                    self.k_means = None
                self.k_means = self.comm.bcast(self.k_means)
                self.Z_local = self.k_means.predict(self.X_local.astype(float))
        else:
            self.k_means = None
            self.Z_local = np.zeros(self.N_p).astype(int)
        self.K += 1
        self.Z_count_local = np.bincount(self.Z_local, minlength = self.K)

    def sample(self):
        total_time = time.time()
        for it in xrange(self.iters):
            start_time = time.time()
            for i in xrange(self.N_p):
                self.sample_Z(i, it)

            if (it % self.L == 0) or (it == max(xrange(self.iters))):
                self.posterior_update3(it)
                current_time = time.time() - self.total_time

                if self.rank == 0:
                    self.predictive_sample(it)
                    iter_time = time.time() - start_time
                    current_time = np.log(time.time() - total_time)
                    self.likelihood_trace[self.L_dict[it]] = [current_time, self.total_likelihood]
                    self.K_trace[self.L_dict[it]]= self.K
                    print("Iteration: %i\tK: %i\tIteration Time: %.2f s.\tPredictive Log Likelihood: %.2f" % (it,self.Z_count_global.nonzero()[0].size, iter_time,self.predictive_likelihood[self.L_dict[it]]))
                    print("Feature Counts: %s\tAlpha: %.2f" % (self.Z_count_global, self.alpha))
                    self.save_files(it)

                if current_time >= self.max_time: # cap duration to 24 hours
                    break


    def sample_Z(self, i, it):
        if (i == self.i_star) and (self.rank == self.P_star):
            self.slice_local[i] = np.copy(self.slice_star)
        else:
            self.slice_local[i] = np.random.uniform(self.slice_star, self.pi[self.Z_local[i]])

        self.Z_count_local[self.Z_local[i]] -= 1
        cluster_likelihood = -np.inf*np.ones(self.K)
        for k in np.where(self.pi >= self.slice_local[i])[0]:
            cluster_likelihood[k] = ((self.X_local[i]+self.prior_gamma)*np.log(self.phi_pi[k])).sum()

#        cluster_likelihood = np.array([((self.X_local[i]+self.prior_gamma)*np.log(self.phi_pi[k])).sum() for k in xrange(self.K)])
#        cluster_likelihood[self.pi < self.slice_local[i]] = -np.inf
        cluster_likelihood -= logsumexp(cluster_likelihood)
        cluster_likelihood = np.exp(cluster_likelihood)

        self.Z_local[i] = np.random.choice(self.K, p=cluster_likelihood)
        self.Z_count_local[self.Z_local[i]] += 1
#        self.obs_likelihood[i] = ((self.X_local[i]+self.prior_gamma)*np.log(self.phi_pi[self.Z_local[i]])).sum()
#        self.obs_likelihood[i] += gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+1).sum()

#        self.obs_likelihood[i] = np.multiply((self.X_local[i] + self.prior_gamma),(np.log(self.phi_pi[self.Z_local[i]]).reshape(-1,self.D))).sum()
#        self.obs_likelihood[i] += gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+ np.ones(self.D)).sum()


        assert(self.Z_count_local.sum() == self.N_p)

    def posterior_update3(self, it):
        self.comm.barrier()
        Z_plus_count_global = self.comm.allreduce(self.Z_count_local)
        nnz_K_plus = Z_plus_count_global.nonzero()[0]
        local_posterior = np.zeros((nnz_K_plus.size,self.D))
        local_dict = {}

        for idx,k in enumerate(nnz_K_plus):
            if self.Z_count_local[k] > 0:
                local_posterior[idx] = self.X_local[np.where(self.Z_local == k)].sum(axis=0).astype(float)
            local_dict[k] = idx

        self.posterior_counts = self.comm.reduce(local_posterior)
#        temp_K, _ = self.posterior_counts.shape
        self.K=nnz_K_plus.size
#        assert(self.phi_pi.shape ==(self.K,self.D))
        self.Z_local = np.array([local_dict[z] for z in self.Z_local]).astype(int)
        self.Z_count_local = np.bincount(self.Z_local, minlength = self.K)
        self.Z_count_global = self.comm.reduce(self.Z_count_local)

#        assert(self.Z_count_global.size==self.K)
        if self.rank == 0:
#            self.K = temp_K
            eta = np.random.beta(self.alpha + 1, self.N)
            pi_eta = self.K / (self.K + (self.N*(1.-np.log(eta))))
            mixture_prob = np.random.uniform()
            if mixture_prob < pi_eta:
                self.alpha = np.random.gamma(1.+self.K, 1./(self.N*(1.-np.log(eta))))
            else:
                self.alpha = np.random.gamma(self.K, 1./(self.N*(1.-np.log(eta))))

            assert(self.posterior_counts.shape == (self.K,self.D))

            self.phi_pi = np.empty(self.posterior_counts.shape)
            for k in xrange(self.K):
                self.phi_pi[k] = np.random.dirichlet(self.posterior_counts[k] + self.prior_gamma)

            temp_Z_count= np.copy(self.Z_count_global).astype(float)
            temp_Z_count=np.append(temp_Z_count,self.alpha)
#            temp_Z_count[temp_Z_count < 1] = self.alpha
            temp_pi = np.random.dirichlet(temp_Z_count)
            self.pi, beta_star = temp_pi[:-1], temp_pi[-1]
            b_k = np.random.beta(1, self.Z_count_global)
            u_star_k  = self.pi*b_k
            min_k = np.argmin(u_star_k)
            self.slice_star = u_star_k[min_k]
            pi_star = np.array([])
            K_star = 0
#            pi_eta /= (pi_eta + (self.N*))

            while beta_star >= self.slice_star:
                K_star +=1
                b_k_star = np.random.beta(1.,self.alpha)
                pi_star = np.append(pi_star,  beta_star * b_k_star)
                beta_star *= (1. - b_k_star)
            if K_star:
                self.Z_count_local = np.concatenate((self.Z_count_local, np.zeros(K_star).astype(int)))
                self.Z_count_global = np.concatenate((self.Z_count_global, np.zeros(K_star).astype(int)))
                self.phi_pi = np.vstack((self.phi_pi,np.random.dirichlet(self.prior_gamma,size=K_star)))
                self.posterior_counts = np.vstack((self.posterior_counts,np.zeros((K_star,self.D))))
                self.pi = np.concatenate((self.pi, pi_star))
                self.K += K_star
        else:
            min_k = None
            self.pi = None
            self.slice_star = None
            self.phi_pi = None
            self.K = None
            self.Z_count_local = None
            self.Z_count_global = None
            self.alpha = None


        min_k = self.comm.bcast(min_k)
        self.K = self.comm.bcast(self.K)
        self.Z_count_local = self.comm.bcast(self.Z_count_local)
        self.Z_count_global = self.comm.bcast(self.Z_count_global)
        self.pi = self.comm.bcast(self.pi)
        self.slice_star = self.comm.bcast(self.slice_star)
        self.posterior_counts = self.comm.bcast(self.posterior_counts)
        self.phi_pi = self.comm.bcast(self.phi_pi)
        self.alpha = self.comm.bcast(self.alpha)

        if self.Z_count_local[min_k] > 0:
            i_p_star = zip(np.where(self.Z_local==min_k)[0],[self.rank]*self.Z_count_local[min_k])
        else:
            i_p_star = [(np.nan, np.nan)]
        gather_i_star = self.comm.gather(i_p_star)

        if self.rank == 0:
            gather_i_star = np.vstack(gather_i_star)
            gather_i_star = gather_i_star[np.where(~np.isnan(gather_i_star))[0]].astype(int)
            N_z,_ =  gather_i_star.shape
            choose_i = np.random.choice(N_z)
            self.i_star, self.P_star = gather_i_star[choose_i]
        else:
            self.i_star = None
            self.P_star = None

        self.i_star = self.comm.bcast(self.i_star)
        self.P_star = self.comm.bcast(self.P_star)

        self.obs_likelihood = np.array([np.multiply((self.X_local[i] + self.prior_gamma),(np.log(self.phi_pi[self.Z_local[i]]).reshape(-1,self.D))).sum() for i in xrange(self.N_p)])
        self.obs_likelihood += np.array([gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+ np.ones(self.D)).sum() for i in xrange(self.N_p)])
        self.total_likelihood = self.comm.reduce(self.obs_likelihood.sum())

    def log_dir_mult(self,X,k):
        a_post = self.posterior_counts[k]
        assert(np.prod(X.shape)==self.D)
        assert(a_post.size==self.D)
        LL = gammaln(X.sum()+1) + gammaln((a_post + self.prior_gamma).sum())
        LL -= gammaln((X + a_post + self.prior_gamma).sum())
        LL += gammaln(X + a_post + self.prior_gamma).sum()
        LL -= gammaln(X+np.ones(self.D)).sum()
        LL -= gammaln(a_post + self.prior_gamma).sum()
        return(LL)

    def predictive_sample(self,it):
        for i_star in xrange(self.N_star):
#            nnz_Z = self.Z_count_global.nonzero()[0]
#            cluster_likelihood = np.array([self.log_dir_mult(self.X_star[i_star],k) for k in nnz_Z])
            cluster_likelihood = np.array([self.log_dir_mult(self.X_star[i_star],k) for k in xrange(self.K)])
#            print("Likelihood shape: "+ str(cluster_likelihood.shape))
#            print("Z_count shape: "+str(self.Z_count_global.shape))
            assert(cluster_likelihood.size == self.Z_count_global.size)
            cluster_likelihood += np.log(self.Z_count_global + self.alpha)
            self.predictive_likelihood[self.L_dict[it]] += logsumexp(cluster_likelihood)

    def save_files(self,it):
        self.today = datetime.datetime.today().strftime("%Y-%m-%d-%f")
        self.fname_foot = self.fname + "_it" + str(it) + "_P" + str(self.P) + "_" + self.today
        if it < max(xrange(self.iters)):
            save_dict = {'likelihood':self.likelihood_trace[:self.L_dict[it]], 'K_trace':self.K_trace[:self.L_dict[it]],
                         'Z_count':self.Z_count_global,'features':self.phi_pi,
                         'pi':self.pi, 'predict_likelihood':self.predictive_likelihood[:self.L_dict[it]],
                         'iters':np.sort(self.L_dict.keys())}
        else:
            save_dict = {'likelihood':self.likelihood_trace, 'K_trace':self.K_trace,
                         'Z_count':self.Z_count_global,'features':self.phi_pi,
                         'pi':self.pi, 'predict_likelihood':self.predictive_likelihood,
                         'iters':np.sort(self.L_dict.keys())}
        savemat(os.path.abspath(self.fname_foot+".mat"),save_dict)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    parser = argparse.ArgumentParser(description="Slice Sampler for Dirichlet Process of Multinomial-Dirichlet Mixtures from Ge et al. (2015)")
    parser.add_argument('--init', type=str, default="dp",
                        help='Initialization type. Valid options are: dp, rand and single')

    parser.add_argument('--data', type=str, default="yale",
                        help='Dataset. Valid options are: cifar, yale and mnist')

    parser.add_argument("-I", "--iters", help="Number of iterations, int.",
                        type=int, default=1000)

    args = parser.parse_args()
    data_type = args.data
    init_type = args.init
    iters = args.iters

    assert(data_type == "yale" or data_type == "mnist" or data_type == "cifar")
    assert(init_type == "dp" or init_type == "rand" or init_type == "single")
    if comm.Get_rank() == 0:
        print("Data type: %s, initialization: %s" % (data_type,init_type))

    # Greyscale CIFAR-10
    if data_type == "cifar":
        cifar_train = np.memmap("../data/cifar", dtype="uint8", mode="r+",shape=(50000L, 1024L))
        cifar_test = np.memmap("../data/cifar_test", dtype="uint8", mode="r+",shape=(10000L, 1024L))
        if init_type == "rand":
        	dp = SliceDP(data=cifar_train, L=5, init_K=200, iters = iters,
                            X_star=cifar_test,
                            rand_init=True, fname="../figs/CIFAR_uncollapsed_rand_init")
        elif init_type == "dp":
        	dp = SliceDP(data=cifar_train, L=5, init_K=200, iters = iters,
                            X_star=cifar_test,
                            rand_init=False, fname="../figs/CIFAR_uncollapsed_DP_init")
        elif init_type=="single":
        	dp = SliceDP(data=cifar_train, L=5, init_K=1, iters = iters,
                            X_star=cifar_test,
                            rand_init=False, fname="../figs/CIFAR_uncollapsed_single_init")
        del cifar_train
        del cifar_test

    # Big MNIST
    elif data_type == "mnist":
        if init_type == "rand":
        	dp = SliceDP(data="../data/big_mnist_train", L=5, init_K=200, iters = iters,
                            X_star="../data/big_mnist_test",
                            rand_init=True, fname="../figs/MNIST_uncollapsed_rand_init")
        elif init_type == "dp":
        	dp = SliceDP(data="../data/big_mnist_train", L=5, init_K=200, iters = iters,
                            X_star="../data/big_mnist_test",
                            rand_init=False, fname="../figs/MNIST_uncollapsed_km_init")
        elif init_type=="single":
        	dp = SliceDP(data="../data/big_mnist_train", L=5, init_K=1, iters = iters,
                            X_star="../data/big_mnist_test",
                            rand_init=False, fname="../figs/MNIST_uncollapsed_single_init")
        # Yale Faces
    elif data_type == "yale":
        data_mat = loadmat(os.path.abspath("../data/extendedYale.mat"))
        if init_type == "rand":
        	dp = SliceDP(data=data_mat['train_data'], L=1, init_K=200, iters = iters,
                            X_star=data_mat['test_data'],
                            rand_init=True, fname="../figs/faces_uncollapsed_rand_init")
        elif init_type == "dp":
        	dp = SliceDP(data=data_mat['train_data'], L=5, init_K=200, iters = iters,
                            X_star=data_mat['test_data'],
                            rand_init=False, fname="../figs/faces_uncollapsed_km_init")
        elif init_type=="single":
        	dp = SliceDP(data=data_mat['train_data'], L=5, init_K=1, iters = iters,
                            X_star=data_mat['test_data'],
                            rand_init=False, fname="../figs/faces_uncollapsed_single_init")
        del data_mat

    dp.sample()
    exit()