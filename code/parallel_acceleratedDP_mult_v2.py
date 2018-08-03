# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:23:21 2018

Parallel Mutltinomial Mixture Model Accelerated DP

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
from scipy.special import gammaln, betaln
from scipy.io import loadmat
from scipy.io import savemat
import mmap
import struct
from array import array as pyarray
from scipy.sparse import lil_matrix
np.random.seed(8888)

class AcceleratedDP(object):

    def __init__(self, data, init_K=5, iters=1000, alpha=100.,
                 M=5, prior_gamma=1., L=5, bin_threshold=2, X_star=None,
                 collapsed=False, rand_init=True,fname="../figs/output",
                 max_time=172800.):
        """
        data: numpy array or string, training data. If numpy array then data is
              NxD file. If string, then the string is the filename of MNIST
              data generated from infimnist.
        init_K: int, number of initial clusters
        iters: int, number of MCMC iterations
        alpha: float, DP concentration parameter
        M: int, number of instantiated new clusters, 'm' in Neal's
           Algorithm 8(2000)
        prior_gamma: float, parameter for Dirichlet prior on multinomial
                     likelihood
        L: int, number of iterations to run before triggering synchronization
           step
        bin_threshold: int, fraction of total iterations to run (i.e.
                        accelerated stage is run for the first
                        (iters / bin_threshold) iterations)
        X_star: numpy array or string, If numpy array then data is
                N_star x D file. If string, then the string is the filename of
                MNIST data generated from infimnist.
        colapsed: bool, if True then run collapsed sampler. Only a valid
                  sampler if run on one processor.
        rand_init: bool, if True then clusters initialized randomly.
        fname: string, directory of where to save output files
        max_time: float, maximum number of time (in seconds) to run sampler,
                  accelerated  stage is automatically stopped after max_time/2
                  seconds.
        """
        self.max_time = float(max_time)
        self.total_time = time.time()
        self.M_local = int(M)
        self.M = int(M)
        self.K = int(init_K)
        self.K_plus = int(init_K)
        self.K_star = 0
        self.fname = str(fname)
        assert(self.K > 0)
        assert(self.M > 0)
        self.iters = int(iters)
        self.H = self.M + self.K
        self.L = int(L)
        self.bin_threshold = int(bin_threshold)
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.rand_init = bool(rand_init)
        assert(self.H >0)
        self.alpha = alpha # parameter for dirichlet mixture
        assert(self.alpha > 0.)
        self.k_means = None
        self.collapsed= bool(collapsed)

#        if data=="../data/big_mnist_train":
#            fname_img = os.path.abspath(data)
#            with open(fname_img, 'r+') as fimg:
#                m = mmap.mmap(fimg.fileno(), 0)
#                magic_nr, self.N, rows, cols = struct.unpack(">IIII", fimg.read(16))
#
#            if self.rank == 0:
#                mnist_idx=np.array_split(xrange(self.N),self.P)
#            else:
#                mnist_idx=None
#            mnist_idx = self.comm.scatter(mnist_idx)
#            self.N_p = int(mnist_idx.size)
#            self.D = int(rows*cols)
#            self.X_local = lil_matrix(np.zeros((self.N_p, self.D), dtype=np.uint8))
#            for idx, i in enumerate(mnist_idx):
#                self.X_local[idx] =  pyarray("B",m[16+(i*self.D) : 16+((i+1)*self.D)])
#            fimg.close()
#            m.close()
#        else:
#            if self.rank==0:
#                self.X = data.astype(int)
#                self.N, self.D = self.X.shape
#                self.X = np.array_split(self.X, self.P)
#
#            else:
#                self.N = None
#                self.D = None
#                self.X = None
#
#            self.N = self.comm.bcast(self.N)
#            self.D = self.comm.bcast(self.D)
#            self.X_local = self.comm.scatter(self.X)
#
#            self.X_local = self.X_local.reshape(-1,self.D).astype(int)
#            self.N_p, D_p = self.X_local.shape
#            assert(D_p == self.D)

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
        self.N_p, D_p = self.X_local.shape
        assert(D_p == self.D)

        self.Z_local = np.random.choice(init_K,size=self.N_p) #np.zeros(self.N_p).astype(int)
        self.pi = np.random.dirichlet([self.alpha]*self.D, size=self.H)
        assert(prior_gamma > 0)
        self.prior_gamma = prior_gamma*np.ones(self.D)

        self.X = None

        if self.rank == 0:
            sync_iters = [it for it in xrange(self.iters) if (it % self.L == 0) or it==max(xrange(self.iters))]
            self.trace_size = len(sync_iters)
            self.L_dict = {it:L for L,it in enumerate(sync_iters)}
            self.K_trace= np.empty(self.trace_size)
            self.likelihood_trace = np.empty((self.trace_size,2))
            self.phi_pi = np.random.dirichlet(self.prior_gamma, size=self.H)
            self.pi = np.random.dirichlet([self.alpha]*self.H)
            self.p_star = np.random.choice(self.P)
            self.X_star = X_star
            if self.X_star is None:
                self.N_star = None
                self.predictive_likelihood = None

            else:
                self.N_star, _ = self.X_star.shape
                self.predictive_likelihood = np.zeros(self.trace_size)


#            if X_star=="../data/big_mnist_test":
#                fname_img = os.path.abspath(X_star)
#                with open(fname_img, 'r+') as fimg:
#                    m = mmap.mmap(fimg.fileno(), 0)
#                    magic_nr, self.N_star, rows, cols = struct.unpack(">IIII", fimg.read(16))
#
#                if self.rank == 0:
#                    mnist_idx=np.array_split(xrange(self.N_star),self.P)
#                else:
#                    mnist_idx=None
#                mnist_idx = self.comm.scatter(mnist_idx)
#                self.X_star = lil_matrix(np.zeros((mnist_idx.size, self.D), dtype=np.uint8))
#                for idx, i in enumerate(mnist_idx):
#                    self.X_local[idx] =  pyarray("B",m[(i*self.D)+16 : 16+((i+1)*self.D)])
#                fimg.close()
#                m.close()
#                del fimg
#                del m
#            else:
#                self.X_star = X_star
#                self.N_star, _ = self.X_star.shape
#                if self.X_star is None:
#                    self.N_star = None
#                    self.predictive_likelihood = None

        else:
            self.phi_pi = None
            self.trace_size = None
            self.L_dict = None
            self.K_trace = None
            self.likelihood_trace = None
            self.X_star = None
            self.N_star = None
            self.predictive_likelihood = None
            self.pi = None
            self.p_star = None

        self.phi_pi = self.comm.bcast(self.phi_pi)
        self.pi = self.comm.bcast(self.pi)
        self.p_star = self.comm.bcast(self.p_star)
        self.total_time = self.comm.bcast(self.total_time)

        self.Z_init()
        self.obs_likelihood = np.array([np.multiply((self.X_local[i] + self.prior_gamma),(np.log(self.phi_pi[self.Z_local[i]]).reshape(-1,self.D))).sum() for i in xrange(self.N_p)])
        self.obs_likelihood += np.array([gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+ np.ones(self.D)).sum() for i in xrange(self.N_p)])
        self.total_likelihood = self.comm.reduce(self.obs_likelihood.sum())

        current_time = time.time() - self.total_time
        self.posterior_update3(0,current_time)#        self.pi = self.comm.bcast(self.pi)


    def Z_init(self):
        if self.K_plus > 1:
            if self.rand_init:
                self.k_means = None
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

        self.Z_count_local = np.bincount(self.Z_local, minlength = self.H)
        self.Z_count_global = self.comm.allreduce(self.Z_count_local)


    def sample(self):
        for it in xrange(self.iters):
            start_time = time.time()

            for i in xrange(self.N_p):
                self.prior_draws(it)
                self.sample_Z(i, it)

            if (it % self.L == 0) or (it == max(xrange(self.iters))):
                current_time = time.time() - self.total_time
                self.posterior_update3(it,current_time)

                if self.rank == 0:
                    self.predictive_sample(it)
                    iter_time = time.time() - start_time
                    self.likelihood_trace[self.L_dict[it]] = [np.log(current_time), self.total_likelihood]
                    self.K_trace[self.L_dict[it]]= self.K
                    print("Iteration: %i\tK: %i\tIteration Time: %.2f s." % (it,self.Z_count_global.nonzero()[0].size, iter_time))
                    print("Accelerated Sampling: %s\tPredictive Log Likelihood: %.2f" % (self.accelerated,self.predictive_likelihood[self.L_dict[it]]))
                    print("Feature Counts: %s\tAlpha: %.2f" % (self.Z_count_global[self.Z_count_global.nonzero()],self.alpha))
                    self.save_files(it)
                else:
                    current_time = None

                self.comm.barrier()
                current_time = self.comm.bcast(current_time)

                if current_time >= self.max_time: # cap duration
                    self.comm.barrier()
                    break


    def prior_draws(self, it):
        Z_local_nnz = self.Z_count_local.nonzero()[0]
        Z_local_zero = np.where(self.Z_count_local==0)[0]
        self.K_star = Z_local_nnz[Z_local_nnz >= self.K_plus].size
        self.K = self.K_plus + self.K_star
        self.empty_cluster = Z_local_zero[Z_local_zero >= self.K_plus]
        self.M_local = self.empty_cluster.size
        assert(self.M_local + self.K  == self.H)
        assert(self.M_local >= 0)
        assert(self.Z_count_local[self.empty_cluster].sum() == 0)
#        current_time= time.time()- self.total_time`
        if self.M_local > 0:
            if self.accelerated:
                data_idx = np.argsort(self.obs_likelihood)[:self.M_local]
                normalized_counts = np.copy(self.X_local[data_idx] + self.prior_gamma)
                normalized_counts /= normalized_counts.sum(axis=1).reshape(self.M_local, -1)
                self.phi_pi[self.empty_cluster] = normalized_counts
            else:
                self.phi_pi[self.empty_cluster] =np.random.dirichlet(self.prior_gamma,size=self.M_local)



    def sample_Z(self, i, it):
        if self.collapsed:
            cluster_likelihood = np.array([self.log_dir_mult(self.X_local[i],k) for k in xrange(self.H)])
        else:
            if self.accelerated:
                cluster_likelihood = np.array([(np.multiply(self.X_local[i]+self.prior_gamma,np.log(self.phi_pi[k]))).sum() for k in xrange(self.H)])
            else:
                if self.rank==self.p_star:
                    cluster_likelihood = np.empty(self.H)
                    cluster_likelihood[:self.K_plus] = np.array([(np.multiply(self.X_local[i]+self.prior_gamma,np.log(self.phi_pi[k]))).sum() for k in xrange(self.K_plus)])
                    cluster_likelihood[self.K_plus:] = np.array([self.log_dir_mult(self.X_local[i],k) for k in xrange(self.K_plus,self.H,1)])
                else:
                    cluster_likelihood = np.array([(np.multiply(self.X_local[i]+self.prior_gamma,np.log(self.phi_pi[k]))).sum() for k in xrange(self.K_plus)])    

        self.Z_count_local[self.Z_local[i]] -= 1
#        prior_cluster_prob = np.zeros(self.H)
        if self.accelerated:
            prior_cluster_prob = np.log(self.Z_count_local)
            if self.M_local:
                prior_cluster_prob[self.empty_cluster] = np.log((self.alpha/self.M_local))
        else:
            if self.P == 1:
                prior_cluster_prob = np.log(self.Z_count_local)
                if self.M_local:
                    prior_cluster_prob[self.empty_cluster] = np.log((self.alpha/self.M_local))

            else:
                if self.rank==self.p_star:
                    prior_cluster_prob = np.zeros(self.H)
                    prior_cluster_prob[:self.K_plus] = np.log(self.pi[:self.K_plus])
                    prior_cluster_prob[self.K_plus:] = np.log(self.Z_count_local[self.K_plus:]/ (self.N - 1. + self.alpha))
                    if self.M_local:
                        prior_cluster_prob[self.empty_cluster] = np.log((self.alpha/self.M_local)/ (self.N - 1. + self.alpha))
                else:
                    prior_cluster_prob = np.copy(np.log(self.pi[:self.K_plus]))
        assert(prior_cluster_prob.size == cluster_likelihood.size)
        cluster_likelihood += prior_cluster_prob
        cluster_likelihood -= logsumexp(cluster_likelihood)
        cluster_likelihood = np.exp(cluster_likelihood)
        self.Z_local[i] = np.random.choice(cluster_likelihood.size, p=cluster_likelihood)

        self.Z_count_local[self.Z_local[i]] += 1

        local_nnz = self.Z_count_local.nonzero()[0]
        nnz_K_star = local_nnz[local_nnz >= self.K_plus]
        self.K_star = nnz_K_star.size
        self.K = self.K_plus + self.K_star

        Z_local_zero = np.where(self.Z_count_local==0)[0]
        self.empty_cluster = Z_local_zero[Z_local_zero >= self.K_plus]
        assert(self.Z_count_local[self.empty_cluster].sum() == 0)
        self.obs_likelihood[i] = np.multiply((self.X_local[i] + self.prior_gamma),(np.log(self.phi_pi[self.Z_local[i]]).reshape(-1,self.D))).sum()
        self.obs_likelihood[i] += gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+ np.ones(self.D)).sum()

        assert(self.Z_count_local.sum() == self.N_p)

    def posterior_update3(self, it,current_time):
        self.comm.barrier()
        Z_plus_count_global = self.comm.allreduce(self.Z_count_local[:self.K_plus])
        nnz_K_plus = Z_plus_count_global.nonzero()[0]
        local_nnz = self.Z_count_local.nonzero()[0]
        nnz_K_star = local_nnz[local_nnz >= self.K_plus]
        new_sizes = self.comm.allgather(self.K_star)

        local_posterior = np.zeros((nnz_K_plus.size,self.D))
        local_counts = np.zeros((self.K_star,self.D))
        local_phi_star = np.zeros((self.K_star,self.D))
        local_dict = {}

        for idx,k in enumerate(nnz_K_plus):
            if self.Z_count_local[k] > 0:
                local_posterior[idx] = self.X_local[np.where(self.Z_local == k)].sum(axis=0).astype(float)
            local_dict[k] = idx

        for idx,k in enumerate(nnz_K_star):
            local_counts[idx] = self.X_local[np.where(self.Z_local == k)].sum(axis=0).astype(float) + self.prior_gamma
            local_phi_star[idx] = np.random.dirichlet(local_counts[idx])
            if self.rank == 0:
                assert(sum(new_sizes[:self.rank]) == 0)
            local_dict[k] = idx + nnz_K_plus.size + sum(new_sizes[:self.rank])

        global_posterior = self.comm.allreduce(local_posterior)

        if self.rank == 0:
            assert(nnz_K_plus.size == global_posterior.shape[0])
            global_phi_plus = np.empty(global_posterior.shape)
            for k in xrange(nnz_K_plus.size):
                global_phi_plus[k] = np.random.dirichlet(global_posterior[k] + self.prior_gamma)
        else:
            global_phi_plus = None

        global_phi_plus = self.comm.bcast(global_phi_plus)
        temp_K_plus,D = global_phi_plus.shape
        global_phi_star = np.vstack(self.comm.allgather(local_phi_star))
        local_counts_star = np.vstack(self.comm.allgather(local_counts))

        temp_K_star, _ = global_phi_star.shape
        self.K_plus = (temp_K_plus + temp_K_star)
        self.K_star = 0
        self.K = (temp_K_plus + temp_K_star)
        self.accelerated=(it < self.iters//self.bin_threshold) and (current_time < self.max_time/2.)
        if self.rank ==0:
            self.alpha = np.random.gamma(self.K, 1. /(1.+0.5772156649+np.log(self.N))) # euler's constant
            self.p_star = np.random.choice(self.P)

        else:
            self.alpha = None
            self.p_star = None
        self.alpha = self.comm.bcast(self.alpha)
        self.p_star = self.comm.bcast(self.p_star)
        self.accelerated = self.comm.bcast(self.accelerated)
#            eta = np.random.beta(self.alpha + 1, self.N)
#            pi_eta = self.K_plus / (self.K_plus + (self.N*(1.-np.log(eta))))
#            mixture_prob = np.random.uniform()
#            if mixture_prob < pi_eta:
#                self.alpha = np.random.gamma(1.+self.K_plus, 1./(self.N*(1.-np.log(eta))))
#            else:
#                self.alpha = np.random.gamma(self.K_plus, 1./(self.N*(1.-np.log(eta))))



        self.H = self.K + self.M
        empty_phi = np.random.dirichlet(self.prior_gamma,size=self.M)
        self.phi_pi = np.vstack((global_phi_plus,global_phi_star,empty_phi))
        zero_counts = np.zeros((self.M,self.D)).astype(int)
        self.posterior_counts = np.vstack((global_posterior,local_counts_star,zero_counts))
        assert(self.posterior_counts.shape == (self.H,self.D))
        assert(self.phi_pi.shape ==(self.H,self.D))
        self.Z_local = np.array([local_dict[z] for z in self.Z_local]).astype(int)
        self.Z_count_local = np.bincount(self.Z_local, minlength = self.H)
        self.Z_count_global = self.comm.allreduce(self.Z_count_local)
        if self.rank == 0:
            self.pi = np.random.dirichlet(self.Z_count_global + self.alpha)
        else:
            self.pi = None
        self.pi = self.comm.bcast(self.pi)
        Z_local_zero = np.where(self.Z_count_local==0)[0]
        self.empty_cluster = Z_local_zero[Z_local_zero >= self.K_plus]
        assert(self.Z_count_local[self.empty_cluster].sum() == 0)
        self.M_local = self.empty_cluster.size

#        self.obs_likelihood = np.array([((self.X_local[i]+self.prior_gamma)*np.log(self.phi_pi[self.Z_local[i]])).sum() for i in xrange(self.N_p)])
#        self.obs_likelihood += np.array([gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+1).sum() for i in xrange(self.N_p)])
        self.obs_likelihood = np.array([np.multiply((self.X_local[i] + self.prior_gamma - 1.),(np.log(self.phi_pi[self.Z_local[i]]).reshape(-1,self.D))).sum() for i in xrange(self.N_p)])
        self.obs_likelihood += np.array([gammaln(self.X_local[i].sum()+1)  - gammaln(self.X_local[i]+ np.ones(self.D)).sum() for i in xrange(self.N_p)])
        self.obs_likelihood += gammaln( sum(self.prior_gamma*self.D ))
        self.obs_likelihood -= sum(gammaln([self.prior_gamma]*self.D))
        self.total_likelihood = self.comm.reduce(self.obs_likelihood.sum())

    def log_dir_mult(self,X,k):
        a_post = self.posterior_counts[k]
        assert(np.prod(X.shape)==self.D)
        assert(a_post.size==self.D)
        LL = gammaln(X.sum()+1) + gammaln((a_post + self.prior_gamma).sum())
        LL -= gammaln(X.sum() + (a_post + self.prior_gamma).sum())
        #product term
        prod_term = gammaln(X + a_post + self.prior_gamma)
        prod_term -= gammaln(X+np.ones(self.D))
        prod_term -= gammaln(a_post + self.prior_gamma)
        prod_term = prod_term.sum()
        LL += prod_term
#        LL += gammaln(X + a_post + self.prior_gamma).sum()
#        LL -= gammaln(X+np.ones(self.D)).sum()
#        LL -= gammaln(a_post + self.prior_gamma).sum()
#        assert(LL2 == LL)
        return(LL)

    def predictive_sample(self,it):
        for i_star in xrange(self.N_star):
            cluster_likelihood = np.array([self.log_dir_mult(self.X_star[i_star],k) for k in xrange(self.H)])
            cluster_likelihood += np.log(self.Z_count_global + self.alpha)
            self.predictive_likelihood[self.L_dict[it]] += logsumexp(cluster_likelihood)


    def save_files(self,it):
        self.today = datetime.datetime.today().strftime("%Y-%m-%d-%f")
        self.fname_foot = self.fname + "_P" + str(self.P)# + "_" + self.today
        if it < max(xrange(self.iters)):
            save_dict = {'likelihood':self.likelihood_trace[:self.L_dict[it]], 'K_trace':self.K_trace[:self.L_dict[it]],
                         'Z_count':self.Z_count_global,'features':self.phi_pi,
                         'pi':self.pi, 'predict_likelihood':self.predictive_likelihood[:self.L_dict[it]],
                         'iters':np.sort(self.L_dict.keys())[:self.L_dict[it]],
                         'collapsed':int(self.collapsed)}
        else:
            save_dict = {'likelihood':self.likelihood_trace, 'K_trace':self.K_trace,
                         'Z_count':self.Z_count_global,'features':self.phi_pi,
                         'pi':self.pi, 'predict_likelihood':self.predictive_likelihood,
                         'iters':np.sort(self.L_dict.keys()),
                         'collapsed':int(self.collapsed)}
        savemat(os.path.abspath(self.fname_foot+".mat"),save_dict)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    parser = argparse.ArgumentParser(description="Accelerate inference code for Multinomial DP mixture from Zhang and Perez-Cruz (2017)")
    parser.add_argument('--init', type=str, default="dp",
                        help='Initialization type. Valid options are: dp, rand and single')

    parser.add_argument('--data', type=str, default="yale",
                        help='Dataset. Valid options are: cifar, yale and mnist')

    parser.add_argument("-I", "--iters", help="Number of iterations, int.",
                        type=int, default=1000)
    parser.add_argument("-K", "--init_K", help="Number of initial clusters",
                        type=int, default=100)
    args = parser.parse_args()
    data_type = args.data
    init_type = args.init
    iters = args.iters
    initial_K = args.init_K

    assert(data_type == "yale" or data_type == "mnist" or data_type == "cifar")
    assert(init_type == "dp" or init_type == "rand" or init_type == "single")
    if comm.Get_rank() == 0:
        print("Data type: %s, initialization: %s" % (data_type,init_type))

    # Greyscale CIFAR-10
    if data_type == "cifar":
        cifar_train = np.memmap("../data/cifar", dtype="uint8", mode="r+",shape=(50000L, 1024L))
        cifar_test = np.memmap("../data/cifar_test", dtype="uint8", mode="r+",shape=(10000L, 1024L))
        if init_type == "rand":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=cifar_train, L=5, init_K=initial_K, iters = iters,
                                collapsed=False, M=50,
                                X_star=cifar_test, bin_threshold=2,
                                rand_init=True, fname="../figs/CIFAR_accelerated_rand_init")
            else:
                dp = AcceleratedDP(data=cifar_train, L=1, init_K=initial_K, iters = iters,
                                collapsed=True, M=50,
                                X_star=cifar_test, bin_threshold=1001,
                                rand_init=True, fname="../figs/CIFAR_collapsed_rand_init")

        elif init_type == "dp":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=cifar_train, L=5, init_K=initial_K, iters = iters,
                                collapsed=False, M=50,
                                X_star=cifar_test, bin_threshold=2,
                                rand_init=False, fname="../figs/CIFAR_accelerated_DP_init")
            else:
            	dp = AcceleratedDP(data=cifar_train, L=1, init_K=initial_K, iters = iters,
                                collapsed=True, M=50,
                                X_star=cifar_test, bin_threshold=1001,
                                rand_init=False, fname="../figs/CIFAR_collapsed_DP_init")
        elif init_type=="single":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=cifar_train, L=5, init_K=1, iters = iters,
                                collapsed=False, M=50,
                                X_star=cifar_test, bin_threshold=2,
                                rand_init=False, fname="../figs/CIFAR_accelerated_single_init")
            else:
                dp = AcceleratedDP(data=cifar_train, L=1, init_K=1, iters = iters,
                                   collapsed=True, M=50,
                                   X_star=cifar_test, bin_threshold=2,
                                   rand_init=False, fname="../figs/CIFAR_collapsed_single_init")
        del cifar_train
        del cifar_test

    #MNIST
    elif data_type == "mnist":
        mnist = loadmat("../data/mnist.mat")
        if init_type == "rand":            
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=mnist['X'], L=5, init_K=initial_K, iters = iters,
                                collapsed=False, M=50,
                                X_star=mnist['X_star'], bin_threshold=2,
                                rand_init=True, fname="../figs/MNIST_accelerated_rand_init")
            else:
                 dp = AcceleratedDP(data=mnist['X'], L=1, init_K=initial_K, iters = iters,
                                collapsed=True, M=50,
                                X_star=mnist['X_star'], bin_threshold=1001,
                                rand_init=True, fname="../figs/MNIST_collapsed_rand_init")

        elif init_type == "dp":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=mnist['X'], L=5, init_K=initial_K, iters = iters,
                                collapsed=False, M=50,
                                X_star=mnist['X_star'], bin_threshold=2,
                                rand_init=False, fname="../figs/MNIST_accelerated_km_init")
            else:
                dp = AcceleratedDP(data=mnist['X'], L=1, init_K=initial_K, iters = iters,
                                   collapsed=True, M=50,
                                   X_star=mnist['X_star'], bin_threshold=1001,
                                   rand_init=False, fname="../figs/MNIST_collapsed_km_init")
        elif init_type=="single":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=mnist['X'], L=5, init_K=1, iters = iters,
                                collapsed=False, M=50,
                                X_star=mnist['X_star'], bin_threshold=2,
                                rand_init=False, fname="../figs/MNIST_accelerated_single_init")
            else:
            	dp = AcceleratedDP(data=mnist['X'], L=1, init_K=1, iters = iters,
                                collapsed=True, M=50,
                                X_star=mnist['X_star'], bin_threshold=1001,
                                rand_init=False, fname="../figs/MNIST_collapsed_single_init")
    # Yale Faces
    elif data_type == "yale":
        data_mat = loadmat(os.path.abspath("../data/extendedYale.mat"))
        if init_type == "rand":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=data_mat['train_data'], L=5, init_K=initial_K, iters = iters,
                                collapsed=False, M=50,
                                X_star=data_mat['test_data'], bin_threshold=2,
                                rand_init=True, fname="../figs/faces_accelerated_rand_init")
            else:
            	dp = AcceleratedDP(data=data_mat['train_data'], L=1, init_K=initial_K, iters = iters,
                                collapsed=True, M=50,
                                X_star=data_mat['test_data'], bin_threshold=1001,
                                rand_init=True, fname="../figs/faces_collapsed_rand_init")
        elif init_type == "dp":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=data_mat['train_data'], L=5, init_K=initial_K, iters = iters,
                                    collapsed=False, M=50,
                                    X_star=data_mat['test_data'], bin_threshold=2,
                                    rand_init=False, fname="../figs/faces_accelerated_km_init")
            else:
                dp = AcceleratedDP(data=data_mat['train_data'], L=1, init_K=initial_K, iters = iters,
                                   collapsed=True, M=50,
                                   X_star=data_mat['test_data'], bin_threshold=1001,
                                rand_init=False, fname="../figs/faces_collapsed_km_init")
        elif init_type=="single":
            if comm.Get_size() > 1:
            	dp = AcceleratedDP(data=data_mat['train_data'], L=5, init_K=1, iters = iters,
                                    collapsed=False, M=50,
                                    X_star=data_mat['test_data'], bin_threshold=2,
                                    rand_init=False, fname="../figs/faces_accelerated_single_init")
            else:
                 dp = AcceleratedDP(data=data_mat['train_data'], L=1, init_K=1, iters = iters,
                                    collapsed=True, M=50,
                                    X_star=data_mat['test_data'], bin_threshold=1001,
                                    rand_init=False, fname="../figs/faces_collapsed_single_init")
        del data_mat

    dp.sample()
