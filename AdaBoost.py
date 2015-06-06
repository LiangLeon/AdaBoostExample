# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 21:29:06 2015

@author: leftword
"""

import numpy as np
import math

class AdaBoost:
    def __init__(self):
        self.max_round = 100
        self.current_round = 0
        self.total_error = 0
    def get_adaboost_model(self, data, desired_error):
        self.data_size = len(data)
        self.data_dim = data[:,0:-1].shape[1]
        self.data_y = data[:,-1]
        #inital weights = [1/N . . .]
        weights = np.linspace(1.0/self.data_size , 1.0/self.data_size , self.data_size)
        self.g_stack = []
        g_x_stack = []
        self.alpha_stack = []
        import DecisionStumps
        Decision_Stumps = DecisionStumps.DecisionStumps()
        while True:
            self.current_round += 1  
            [g,g_x,error] = Decision_Stumps.get_g_and_predect_result(self.data_size, self.data_dim, data, weights)
            delta = math.sqrt((1-error)/error)
            self.g_stack.append(g)
            g_x_stack.append(g_x)
            self.alpha_stack.append(math.log1p(delta))
        
            for index, (g_x_i, y) in enumerate(zip(g_x,self.data_y)):
                if g_x_i != y:
                    weights[index] *= delta
                else:
                    weights[index] /= delta
            #After re-weights, normalize to distribution 
            weights /= sum(weights)
            total = np.array([])
            for alpha, g_result in zip(self.alpha_stack,g_x_stack):
                temp = np.array([i * alpha for i in g_result])
                if np.array_equal(total,[]):
                    total = temp
                else:
                    total += temp

            for index, answer in enumerate(total):
                if answer >= 0:
                    total[index] = 1
                else:
                    total[index] = -1
            
            self.total_error = 0
            for predict_result, y in zip(total,self.data_y):
                if predict_result != y:
                    self.total_error += 1
            
            self.total_error /= float(self.data_size)
            if self.total_error <= desired_error or self.current_round >= self.max_round:
                break
        return self.g_stack, self.alpha_stack
            
        