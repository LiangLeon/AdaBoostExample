# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 21:29:06 2015

@author: leftword
"""

import numpy as np
import math

class AdaBoost:
    def __init__(self):
        pass
    def get_adaboost_model(self, data, desired_error, plot_result=None):
        max_round = 100
        current_round = 0
        total_error = 0
        data_size = len(data)
        data_dim = data[:,0:-1].shape[1]
        data_y = data[:,-1]
        #inital weights = [1/N . . .]
        weights = np.linspace(1.0/data_size , 1.0/data_size , data_size)
        g_stack = []
        g_x_stack = []
        alpha_stack = []
        import DecisionStumps
        Decision_Stumps = DecisionStumps.DecisionStumps()
        while True:
            current_round += 1  
            [g,g_x,error] = Decision_Stumps.get_g_and_predect_result(data_size, data_dim, data, weights)
            delta = math.sqrt((1-error)/error)
            g_stack.append(g)
            g_x_stack.append(g_x)
            alpha_stack.append(math.log1p(delta))
        
            for index, (g_x_i, y) in enumerate(zip(g_x,data_y)):
                if g_x_i != y:
                    weights[index] *= delta
                else:
                    weights[index] /= delta
            #After re-weights, normalize to distribution 
            weights /= sum(weights)
            total = np.array([])
            for alpha, g_result in zip(alpha_stack,g_x_stack):
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
            
            total_error = 0
            for predict_result, y in zip(total,data_y):
                if predict_result != y:
                    total_error += 1
            
            total_error /= float(data_size)
            if total_error <= desired_error or current_round >= max_round:
                break
            
        final_g_stack = []
        final_alpha_stack = []
            
        for _g, _a in zip(g_stack,alpha_stack):
            if _g not in final_g_stack:
                final_g_stack.append(_g)
                final_alpha_stack.append(_a)
            else:
                final_alpha_stack[final_g_stack.index(_g)] += _a
        
        if plot_result == True:
            import matplotlib.pyplot as plt
            plt.rcParams['figure.figsize'] = 15, 10
            positive = data[data[:,-1]==1]
            negative = data[data[:,-1]!=1]
            xmin, xmax = data[:,0].min()-0.1, data[:,0].max()+0.1
            ymin, ymax = data[:,1].min()-0.1, data[:,1].max()+0.1
            xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1))
            xnew = np.c_[xx.ravel(), yy.ravel()]
    
            ynew = self.predict_results(final_g_stack,final_alpha_stack,xnew).reshape(xx.shape)
            plt.figure(1)
            plt.set_cmap(plt.cm.Blues)
            plt.pcolormesh(xx, yy, ynew)
            plt.plot(positive[:,0], positive[:,1], 'ob',markersize=10)
            plt.plot(negative[:,0], negative[:,1], 'xr',markersize=16)
            plt.show()
            
        return final_g_stack, final_alpha_stack
        
    def predict_result(self, g_s, a_s, data):
        result = 0
        for g, alpha in zip(g_s, a_s):
            th = g[0]
            dim = g[1]
            k = g[2]
            if k * ( data[dim] - th ) >= 0:
                result += alpha
            else:
                result -= alpha
        
        if result >= 0:
            return 1
        else:
            return -1

    def predict_results(self, g_s, a_s, data):
        result = np.zeros(len(data))
        for i,_data in enumerate(data):
            result[i] = self.predict_result(g_s,a_s,_data)
        return result
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
        