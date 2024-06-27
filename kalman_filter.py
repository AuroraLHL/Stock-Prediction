'''
FilePath: /大作业/kalman_filter.py
Author: LHL
Date: 2022-05-26 16:32:36
LastEditTime: 2022-06-03 13:35:41
LastEditors: LHL
Copyright 2022 by LHL, All Rights Reserved. 
'''
from pykalman import KalmanFilter


def Kalman1D(observations,damping=0.001):
    observation_covariance=damping         #测量值误差
    initial_value_guess=observations[0]    #初始
    transition_matrix=1                    #状态转换矩阵
    transition_covariance=0.1              #系统误差
    kf=KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance, 
        observation_covariance=observation_covariance, 
        transition_covariance=transition_covariance, 
        transition_matrices=transition_matrix
    )
    pred_state,state_cov=kf.smooth(observations)
    
    return pred_state