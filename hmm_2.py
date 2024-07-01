'''
Author: Hongliang Lu, lhl@pku.edu.cn
Date: 2024-07-01 22:04:11
LastEditTime: 2024-07-01 22:22:38
FilePath: /scr/hmm_2.py
Description: 
Organization: College of Engineering,Peking University.
'''

import numpy as np
from math import pi,sqrt,exp,pow
from numpy.linalg import det, inv
from sklearn import cluster
from tqdm import tqdm
import time

"""高斯隐马尔可夫模型"""


# 二元高斯分布函数
def gauss2D(x, mean, cov):
    # x, mean, cov均为numpy.array类型
    z = -np.dot(np.dot((x-mean).T,inv(cov)),(x-mean))/2.0
    temp = pow(sqrt(2.0*pi),len(x))*sqrt(det(cov))
    return (1.0/temp)*exp(z)

class GaussianHMM:
    
    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.x_size = x_size                                                  #输入x的维度
        self.start_prob = np.ones(n_state) * (1.0 / n_state)                  # 初始状态概率
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)    # 状态转换概率矩阵
        self.trained = False                                                  # 是否需要重新训练
        self.n_iter = iter                                                    # EM训练的迭代次数
        self.observe_mean = np.zeros((n_state, x_size))                       # 高斯分布的观测概率均值
        self.observe_vars = np.zeros((n_state, x_size, x_size))               # 高斯分布的观测概率协方差
        for i in range(n_state): 
            self.observe_vars[i] = np.random.randint(0,10)                    # 初始化为均值为0，方差为1的高斯分布函数
    
    """通过K均值聚类，确定状态初始值"""
    def _init(self,X):   #私有方法
        
        mean_kmeans = cluster.KMeans(n_clusters=self.n_state)   #聚类种类数为隐状态数
        mean_kmeans.fit(X)
        self.observe_mean = mean_kmeans.cluster_centers_        #聚类中心作为初始高斯分布的均值
        for i in range(self.n_state):
            self.observe_vars[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))
        print("聚类初始化成功！")
        
    # 求前向概率
    def forward(self, X, Z):    
        """前向算法
        Args:
            X :观测序列
            Z : 隐状态
        Returns:
            : _description_
        """
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.observe_prob(X[0]) * self.start_prob * Z[0] # 初始值
        
        # 归一化因子
        S_alpha=np.zeros(X_length)
        S_alpha[0]=1/np.max(alpha[0])
    
        alpha[0] = alpha[0] * S_alpha[0]
        
        # 递归
        for i in range(X_length):
            if i == 0:
                continue
            alpha[i] = self.observe_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            S_alpha[i]=1/np.max(alpha[i])
            if S_alpha[i]==0:
                continue
            alpha[i] = alpha[i] * S_alpha[i]    #归一化

        return alpha,S_alpha

    # 求后向概率
    def backward(self, X, Z):
        X_length = len(X)
        
        beta = np.zeros((X_length, self.n_state))  
        beta[X_length - 1] = np.ones((self.n_state))
        
        #归一化
        S_beta=np.zeros(X_length)
        S_beta[X_length-1]=np.max(beta[X_length - 1])
        beta[X_length - 1]=beta[X_length - 1]/S_beta[X_length-1]
        
        # 递归
        for i in reversed(range(X_length)):
            if i == X_length - 1:
                continue
            beta[i] = np.dot(beta[i + 1] * self.observe_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            S_beta[i]=np.max(beta[i])
            if S_beta[i]==0:
                continue
            beta[i] = beta[i] / S_beta[i]

        return beta
    
    def observe_prob(self, x): # 求当前X在各个状态下的观测概率 P(X|i)
        prob = np.zeros((self.n_state))
        for i in range(self.n_state):
            prob[i]=gauss2D(x,self.observe_mean[i],self.observe_vars[i])
        return prob
    
    def observe_prob_updated(self, X, post_state): # 更新观测概率
        
        for k in range(self.n_state):
            for j in range(self.x_size):
                self.observe_mean[k][j] = np.sum(post_state[:,k] *X[:,j]) / np.sum(post_state[:,k])

            X_cov = np.dot((X-self.observe_mean[k]).T, (post_state[:,k]*(X-self.observe_mean[k]).T).T)
            self.observe_vars[k] = X_cov / np.sum(post_state[:,k])
            
            if det(self.observe_vars[k]) == 0: # 对奇异矩阵的处理
                self.observe_vars[k] = self.observe_vars[k] + 0.01*np.eye(len(X[0]))  #加上一点扰动
    
    # 针对于单个长序列的训练
    def train(self, X, Z_seq=np.array([])):
        """baum-welch算法
        Args:
            X : 输入X类型：array，数组的形式
            Z_seq :输入Z类型: array，一维数组的形式，默认为空列表（即未知隐状态情况）
        """
        self.trained = True
        X_length = len(X)
        self._init(X)      #初始化参数
        print("开始训练")
        
        start_time=time.time()
        # 状态序列预处理
        # 判断是否已知隐藏状态
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        
        self.L=[]
        for e in tqdm(range(self.n_iter)):  # EM步骤迭代
            # 中间参数
            #print(f"epoch: {e}|{self.n_iter}")
            
            # E步骤
            # 向前向后传递因子
            alpha,S_alpha = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z)  # P(x|z)

            L=np.log(np.sum(alpha[-1]))-np.sum(np.log(S_alpha))
        
            self.L.append(L)
            
            post_state = alpha * beta/(np.sum(alpha*beta,axis=1)).reshape(-1,1)  
            
            post_adj_state = np.zeros((self.n_state, self.n_state))  # 相邻状态的联合后验概率
            for i in range(X_length):
                if i == 0:
                    continue
                now_post_adj_state=np.outer(alpha[i - 1],beta[i]*self.observe_prob(X[i]))*self.transmat_prob
                post_adj_state += now_post_adj_state/np.sum(now_post_adj_state)

            # M步骤，估计参数
            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.n_state):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.observe_prob_updated(X, post_state)  #更新观测概率

        total_time=time.time()-start_time
        print(f"训练完成,耗时：{round(total_time,2)}sec")
        
    def predict(self,origin_X,t):
        """预测直到t+1时刻的值
        Args:
            origin_X :观测值
            t : 想要预测的时刻
        Returns:
            _type_: t时刻之前的所有预测值
        """
        X=origin_X[:t]  
        X_length=len(X)
        Z=np.ones((X_length, self.n_state))
        alpha,_=self.forward(X,Z)
        
        post_state=alpha/(np.sum(alpha,axis=1)).reshape(-1,1)
        now_post_state=post_state
        x_pre=0
        for state in range(self.n_state):
            p_state=now_post_state[:,state]   #后验概率
            temp=0
            for next_state in range(self.n_state):
                temp+=self.observe_mean[next_state]*self.transmat_prob[state][next_state]
            x_pre+=p_state*temp
        
        return x_pre

    def predict_more(self,origin_X,t):
        X=origin_X.copy()
        X_length=len(X)
        
        while(X_length<t):
            Z=np.ones((X_length, self.n_state))
            alpha,_=self.forward(X,Z)
            post_state=alpha/(np.sum(alpha,axis=1)).reshape(-1,1)
            now_post_state=post_state
            x_pre=0
            for state in range(self.n_state):
                p_state=now_post_state[:,state]
                temp=0
                for next_state in range(self.n_state):
                    temp+=self.observe_mean[next_state]*self.transmat_prob[state][next_state]
                x_pre+=p_state*temp
            
            X=np.concatenate([X,x_pre[-1].reshape(-1,1)]) #最后一个值加入
            X_length+=1
        
        return X