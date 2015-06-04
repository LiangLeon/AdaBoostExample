# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:32 2015

@author: leftword
base classifier algorithm
"""
import numpy as np

class DecisionStumps:
    def __init__(self):
        self.data_size = 0
        self.data_dim = 0

    def get_hypothesis(self, data_size, data_dim, data, weights):
        self.data_size = data_size
        self.data_dim = data_dim
        self.data = data
        self.weights = weights

        hypothesis = []
        for i in range(self.data_dim):
            min_error = -1 #min error count
            min_index = -1 #positive or negative
            min_dim = -1   #feature dimension for min error
            min_th = -1    #decision boundary
            arg = np.argsort(self.data,0)[:,i]
            data_feature = self.data[arg,i]
            data_y = self.data[arg,-1]
            data_weights = self.weights[arg]
            decision_boundary = []
            for j in data_feature:
                if j not in decision_boundary:
                    decision_boundary.append(j)
            decision_boundary.append(data_feature[-1]+1)
            
            for j, n in enumerate(decision_boundary):
                boundary = 0
                if n != decision_boundary[-1]:
                    boundary = (decision_boundary[j] + decision_boundary[j+1])/2.0
                else:
                    break
                for k in [1,-1]: #1 for right positive -1 for right negative
                        error_count = 0
                        for feature, y, mu in zip( data_feature, data_y, data_weights ):
                            if k * ( feature - boundary ) >= 0:
                                predict_answer = 1
                            else:
                                predict_answer = -1
                            if predict_answer != y:
                                error_count += mu
                        if min_error == -1:
                            min_error = error_count
                            min_index = k
                            min_dim = i
                            min_th = boundary
                        elif error_count < min_error:
                            min_error = error_count
                            min_index = k
                            min_dim = i
                            min_th = boundary
            hypothesis.append([min_error, min_index, min_dim, min_th])
        return hypothesis
        
    def get_g_and_predect_result(self, data_size, data_dim, data, weights):
        hypothesis = self.get_hypothesis(data_size,data_dim,data,weights)
        dim = 0
        th = 0
        index = 0
        error = -1
        for h in hypothesis:
            if error == -1:
                error = h[0]
                index = h[1]
                dim = h[2]
                th = h[3]
            elif h[0] < error:
                error = h[0]
                index = h[1]
                dim = h[2]
                th = h[3] 
        data_feature = self.data[:,dim]
        data_y = self.data[:,-1]
        data_g_x = []
        for feature, y in zip(data_feature, data_y):
            if index * ( feature - th ) >= 0:
                data_g_x.append(1)
            else:
                data_g_x.append(-1)

        return [[th, dim, index],data_g_x,error]






















           