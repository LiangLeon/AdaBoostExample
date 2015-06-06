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
            decision_boundary_data_count = {}
            for j in data_feature:
                if j not in decision_boundary:
                    decision_boundary.append(j)
                    decision_boundary_data_count[j] = 1
                else:
                    decision_boundary_data_count[j] += 1
            decision_boundary.append(data_feature[-1]+1)

            boundary = (decision_boundary[0] + decision_boundary[1])/2.0

            previous_decision_result = []
            for k in [1,-1]: #1 for right positive -1 for right negative
                error_count = 0
                for feature, y, mu in zip( data_feature, data_y, data_weights ):
                    if k * ( feature - boundary ) >= 0:
                        predict_answer = 1
                    else:
                        predict_answer = -1
                    if predict_answer != y:
                        error_count += mu
                previous_decision_result.append([k, error_count])
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

            data_index = decision_boundary_data_count[decision_boundary[0]]
            for boundary_i in range(1, len(decision_boundary)-1):
                boundary = (decision_boundary[boundary_i] + decision_boundary[boundary_i+1])/2.0
                
                
                pre_d_0 = previous_decision_result[0]
                pre_d_1 = previous_decision_result[1]
                
                pre_k = pre_d_0[0]
                pre_error = pre_d_0[1]
                    
                inner_data_index = 0
                while inner_data_index < decision_boundary_data_count[decision_boundary[boundary_i]]:
                        
                    if pre_k * ( data_feature[data_index + inner_data_index] - boundary ) >= 0:
                        predict_answer = 1
                    else:
                        predict_answer = -1
                    if predict_answer != data_y[data_index + inner_data_index]:
                        pre_error += ( data_weights[data_index + inner_data_index] )
                        pre_d_0[1] = pre_error
                    else:
                        pre_error -= ( data_weights[data_index + inner_data_index] )
                        pre_d_0[1] = pre_error
                    inner_data_index += 1
                
                pre_d_1[1] = 1 - pre_d_0[1]

                if pre_d_0[1] < min_error:
                    min_error = pre_d_0[1]
                    min_index = 1
                    min_dim = i
                    min_th = boundary
                elif pre_d_1[1] < min_error:
                    min_error = pre_d_1[1]
                    min_index = -1
                    min_dim = i
                    min_th = boundary
                    
                data_index += decision_boundary_data_count[decision_boundary[boundary_i]]
                        

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






















           