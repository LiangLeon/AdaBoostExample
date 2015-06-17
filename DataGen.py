# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:21:33 2015

@author: leftword
"""

import numpy as np
import math
import matplotlib.pyplot as plt

class DataGenerator:
    data_dim = 0
    data_size = 0
    def __init__(self):
        pass

    def get_default_data_1(self, plot=None):
        self.data_dim = 2
        data = np.array([[0,0,1], [2,2,1], [-1,-1,1], [2,-3,1], [3,-3,1], [1,-1,-1],[1,-2,-1], [3,3,-1], [3,0,-1],[4,0,-1]])
        self.data_size = len(data)

        if plot == True:
            #self.plot_2d_data(data)
            labels = [1.0/self.data_size] * self.data_size;
            plt.rcParams['figure.figsize'] = 8, 6
            positive = data[data[:,-1]==1]
            negative = data[data[:,-1]!=1]
            xmin, xmax = data[:,0].min()-0.1, data[:,0].max()+0.1
            ymin, ymax = data[:,1].min()-0.1, data[:,1].max()+0.1        
            plt.figure(1)
            plt.set_cmap(plt.cm.Blues)
            axes = plt.gca()
            axes.set_xlim([xmin,xmax])
            axes.set_ylim([ymin,ymax])
            plt.plot(positive[:,0], positive[:,1], 'ob',markersize=10)
            plt.plot(negative[:,0], negative[:,1], '^r',markersize=10)
#            for label, x, y in zip(labels, data[:, 0], data[:, 1]):
#                plt.annotate(
#                label, 
#                xy = (x, y), xytext = (-10, 10),
#                textcoords = 'offset points', ha = 'right', va = 'bottom',
#                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#                arrowprops = None)
            plt.show()
        return data

    def get_default_data_2(self, plot=None):
        self.data_dim = 2
        data = np.array([[0.5,-2.5,1], [3,1,1], [-1.5,-1,-1], [1,3.5,-1]])
        self.data_size = len(data)
    
        if plot == True:
            self.plot_2d_data(data)

        return data

    def get_default_data_3(self, plot=None):
        self.data_dim = 2
        data = np.array([[0,1,1], [-1,0,1], [-1.1,-2,1], [-2,2,1], [-1.5,4,1], [0,5,1], [1,3,1], [1.5,4,1],
                         [2,-3,-1], [-2,-3,-1], [2,1,-1], [3.1,3,-1], [3,5,-1], [1,6,-1], [-2,5,-1], [-3,3,-1]])
        self.data_size = len(data)
        
        if plot == True:
            self.plot_2d_data(data)

        return data
        
    def separable_2d(self, seed, n_points, vect_w=None, plot=None):
        np.random.seed(seed)
        
        if vect_w == None:
            vect_w = [0.5,-0.3]
        else:
            vect_w = vect_w

        dim_x = 2
        data_dim = dim_x + 1 # leading 1 and class value
        data = np.ones(data_dim * n_points).reshape(n_points, data_dim)
 
        # fill in random values
        data[:, 0] = -1 + 2*np.random.rand(n_points)
        data[:, 1] = -1 + 2*np.random.rand(n_points)

        for idx in range(n_points):
            if sum(p * q for p, q in zip(vect_w, data[idx])) >= 0:
                data[idx,-1] = 1
            else:
                data[idx,-1] = -1

        if plot == True:
            self.plot_2d_data(data)
            
        return data
        
    def separable_2d_circle(self, seed, n_points, vect_w=None, plot=None):
        np.random.seed(seed)
        
        if vect_w == None:
            vect_w = [1.5, 0.3, 1]
        else:
            vect_w = vect_w

        dim_x = 2
        data_dim = dim_x + 1 # leading 1 and class value
        data = np.ones(data_dim * n_points).reshape(n_points, data_dim)
        
        x_r = vect_w[0] + vect_w[2] + 0.5
        x_l = vect_w[0] - vect_w[2] - 0.5
        x_range = x_r - x_l
        x_middle = (x_range/2) - vect_w[0]
        
        y_r = vect_w[1] + vect_w[2] + 0.5
        y_l = vect_w[1] - vect_w[2] - 0.5
        y_range = y_r - y_l
        y_middle = (y_range/2) - vect_w[1]
 
        # fill in random values
        data[:, 0] = -x_middle + x_range*np.random.rand(n_points)
        data[:, 1] = -y_middle + y_range*np.random.rand(n_points)

        for idx in range(n_points):
            p = data[idx]
            
            temp = math.sqrt( ( p[0] - vect_w[0] ) *  ( p[0] - vect_w[0] ) + ( p[1] - vect_w[1] ) *  ( p[1] - vect_w[1] ) )
            if temp <= vect_w[2]:
                p[-1] = 1
            else:
                p[-1] = -1

        if plot == True:
            self.plot_2d_data(data)
    
        return data
        
    def plot_2d_data(self, data):
        plt.rcParams['figure.figsize'] = 16, 12
        positive = data[data[:,-1]==1]
        negative = data[data[:,-1]!=1]
        xmin, xmax = data[:,0].min()-0.1, data[:,0].max()+0.1
        ymin, ymax = data[:,1].min()-0.1, data[:,1].max()+0.1        
        plt.figure(1)
        plt.set_cmap(plt.cm.Blues)
        axes = plt.gca()
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        plt.plot(positive[:,0], positive[:,1], 'ob',markersize=20)
        plt.plot(negative[:,0], negative[:,1], '^r',markersize=20)
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        