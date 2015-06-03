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
            min_error = -1 #min error count
            min_index = -1 #positive or negative
            min_dim = -1   #feature dimension for min error
            min_th = -1
            arg = np.argsort(self.data,0)[:,i]
            data_feature = self.data[arg,i]
            data_y = self.data[arg,-1]
            decision_boundary = []
            for j in data_feature:
                if j not in decision_boundary:
                    decision_boundary.append(j)
            decision_boundary.append(data_feature[-1]+1)
            
            for j, n in enumerate(decision_boundary):
                boundary = 0
                if n != decision_boundary[-1]:
                    boundary = (decision_boundary[j] + decision_boundary[j+1])/2.0
                for k in range(2):
                    if k == 0:
                        error_count_p = 0
                        #right is positive
                        for feature, y in zip( data_feature, data_y ):
                            if feature - boundary >= 0:
                                predict_answer = 1
                            else:
                                predict_answer = -1
                            if predict_answer != y:
                                error_count_p += 1
                        if min_error == -1:
                            min_error = error_count_p
                            min_index = k
                            min_dim = i
                            min_th = boundary
                        elif error_count_p < min_error:
                            min_error = error_count_p
                            min_index = k
                            min_dim = i
                            min_th = boundary
                    else:
                        error_count_n = 0
                        #right is negative
                        for feature, y in zip( data_feature, data_y ):
                            if boundary - feature >= 0:
                                predict_answer = 1
                            else:
                                predict_answer = -1
                            if predict_answer != y:
                                error_count_n += 1
                        if min_error == -1:
                            min_error = error_count_n
                            min_index = k
                            min_dim = i
                            min_th = boundary
                        elif error_count_n < min_error:
                            min_error = error_count_n
                            min_index = k
                            min_dim = i
                            min_th = boundary
        return min_error, min_index, min_dim, min_th
            