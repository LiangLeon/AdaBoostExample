# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:32 2015

@author: leftword
base classifier algorithm
"""
import numpy as np

class DecisionStumps:
    def __init__(self, data_size, data_dim, data, weights):
        self.data_size = data_size
        self.data_dim = data_dim
        self.data = data
        self.weights = weights
        self.training_set = []
        for row in data:
            self.training_set.append( (tuple(row[0:2]), row[2]) )
        
    def get_hypothesis(self):
        
        for i in range(self.data_dim):
            arg = np.argsort(self.data,0)[:,i]
            data_feature = self.data[arg,i]            
            feature_min = data_feature[:,i].min()            
            feature_max = data_feature[:,i].max() 
        return feature_min, feature_max
            