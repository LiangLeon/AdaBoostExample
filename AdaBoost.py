# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 21:29:06 2015

@author: leftword
"""

import numpy as np
import math
import matplotlib.pyplot as plt

class AdaBoost:
    def __init__(self):
        pass
    def get_adaboost_model(self, data, desired_error, plot_result=None, plot_g=None):
        max_round = 1000
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

            plt.rcParams['figure.figsize'] = 8, 6
            positive = data[data[:,-1]==1]
            negative = data[data[:,-1]!=1]
            xmin, xmax = data[:,0].min()-0.1, data[:,0].max()+0.1
            ymin, ymax = data[:,1].min()-0.1, data[:,1].max()+0.1
            xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
            xnew = np.c_[xx.ravel(), yy.ravel()]
    
            ynew = self.predict_results(final_g_stack,final_alpha_stack,xnew).reshape(xx.shape)
            plt.figure(1)
            plt.set_cmap(plt.cm.Blues)
            axes = plt.gca()
            axes.set_xlim([xmin,xmax])
            axes.set_ylim([ymin,ymax])
            plt.pcolormesh(xx, yy, ynew)
            plt.plot(positive[:,0], positive[:,1], 'ob',markersize=10)
            plt.plot(negative[:,0], negative[:,1], '^r',markersize=10)
            if plot_g == True:
                for g in final_g_stack:
                    if g[1] == 0:
                        plt.plot([g[0],g[0]],[ymin-1,ymax+1],color='grey', linestyle='-', linewidth=2)
                    elif g[1] == 1: 
                        plt.plot([xmin-1,xmax+1], [g[0],g[0]],color='grey', linestyle='-', linewidth=2)
            plt.show()
            
        return final_g_stack, final_alpha_stack, total_error, current_round
        
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
        
    def ADA_boost_show_demo_plot(self, data, g_s, a_s):
        import decimal
        data_size = len(data)
        #labels = [ decimal.Decimal(1.0/data_size).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN)] * data_size;
        weights = np.linspace(1.0/data_size , 1.0/data_size , data_size)
        labels_min = min(weights)     
        plt.rcParams['figure.figsize'] = 16, 12
        xmin, xmax = data[:,0].min()-0.1, data[:,0].max()+0.1
        ymin, ymax = data[:,1].min()-0.1, data[:,1].max()+0.1    
        xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
        xnew = np.c_[xx.ravel(), yy.ravel()]
        plt.figure(1)
        axes = plt.gca()
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        for label, x, y, ans in zip(weights, data[:, 0], data[:, 1], data[:,-1]):
            m_size = 20 + ( label - labels_min ) * 30
            if m_size <= 0:
                m_size = 1
            if ans == 1:
                plt.plot(x,y,'ob',markersize=m_size,markeredgewidth=2)
            else:
                plt.plot(x,y,'^r',markersize=m_size,markeredgewidth=2)  
            label = decimal.Decimal(label).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN)

#            plt.annotate(
#            label, 
#            xy = (x, y), xytext = (-20, 20),
#            textcoords = 'offset points', ha = 'right', va = 'bottom',
#            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 1.),
#            arrowprops = None,
#            size = 20)
        plt.show()
        if g_s == [] and a_s == []:
            plt.show()
        else:
            inn_g = []
            inn_a = []
            for _g, _a in zip(g_s,a_s):
                inn_g.append(_g)
                inn_a.append(_a)
                ynew = self.predict_results(inn_g,inn_a,xnew).reshape(xx.shape)
                plt.figure(1)
                plt.set_cmap(plt.cm.bwr_r)
                axes = plt.gca()
                axes.set_xlim([xmin,xmax])
                axes.set_ylim([ymin,ymax])
                plt.pcolormesh(xx, yy, ynew)
                error_count = 0
                data_correct = [1] * data_size
                delta = 0.1            
                for g in inn_g:
                    if g[1] == 0:
                        if g == inn_g[-1]:
                            plt.plot([g[0],g[0]],[ymin-1,ymax+1],color='black', linestyle='-', linewidth=5)
                        else:
                            plt.plot([g[0],g[0]],[ymin-1,ymax+1],color='grey', linestyle='-', linewidth=5)                       
                    elif g[1] == 1: 
                        if g == inn_g[-1]:
                            plt.plot([xmin-1,xmax+1], [g[0],g[0]],color='black', linestyle='-', linewidth=5)
                        else:
                            plt.plot([xmin-1,xmax+1], [g[0],g[0]],color='grey', linestyle='-', linewidth=5)                     
                
                for i, _data, in enumerate(data):
                    if self.predict_result([_g], [_a], _data) != _data[-1]:
                        error_count += 1
                        data_correct[i] = 0
                error = error_count / float(data_size)
                if error != 0:
                    delta = math.sqrt((1-error)/error)      
                    for w_index, (w, correct) in enumerate(zip(weights, data_correct)):
                        if correct != 1:
                            weights[w_index] = w * delta
                        else:
                            weights[w_index] = w / delta
                    weights /= sum(weights)
                    for label, x, y, ans in zip(weights, data[:, 0], data[:, 1], data[:,-1]):
                        m_size = 20 + ( label - labels_min ) * 30
                        if m_size <= 0:
                            m_size = 1
                        if ans == 1:
                            plt.plot(x,y,'ob',markersize=m_size,markeredgewidth=2)
                        else:
                            plt.plot(x,y,'^r',markersize=m_size,markeredgewidth=2)  
                        label = decimal.Decimal(label).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN)
#                        plt.annotate(
#                        label, 
#                        xy = (x, y), xytext = (-20, 20),
#                        textcoords = 'offset points', ha = 'right', va = 'bottom',
#                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 1.),
#                        arrowprops = None,
#                        size = 20)   
                    plt.show()
                else:
                    for label, x, y, ans in zip(weights, data[:, 0], data[:, 1], data[:,-1]):
                        m_size = 20 + ( label - labels_min ) * 30
                        if m_size <= 0:
                            m_size = 1
                        if ans == 1:
                            plt.plot(x,y,'ob',markersize=m_size,markeredgewidth=2)
                        else:
                            plt.plot(x,y,'^r',markersize=m_size,markeredgewidth=2)  
                        label = decimal.Decimal(label).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN)
#                        plt.annotate(
#                        label, 
#                        xy = (x, y), xytext = (-20, 20),
#                        textcoords = 'offset points', ha = 'right', va = 'bottom',
#                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 1.),
#                        arrowprops = None,
#                        size = 20)   
                    plt.show()
            
                
                
                
                
                
                
                
                
                
                
                
                
            
        