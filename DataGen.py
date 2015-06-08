# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:21:33 2015

@author: leftword
"""

import numpy as np
import math

class DataGenerator:
    data_dim = 0
    data_size = 0
    def __init__(self):
        pass

    def get_default_data_1(self):
        self.data_dim = 2
        data = np.array([[0,0,1], [2,2,1], [-1,-1,1], [2,-3,1], [3,-3,1], [1,-1,-1],[1,-2,-1], [3,3,-1], [3,0,-1],[4,0,-1]])
        self.data_size = len(data)
        return data

    def get_default_data_2(self):
        self.data_dim = 2
        data = np.array([[0.5,-2.5,1], [3,1,1], [-1.5,-1,-1], [1,3.5,-1]])
        self.data_size = len(data)
        return data

    def get_default_data_3(self):
        self.data_dim = 2
        data = np.array([[0,1,1], [-1,0,1], [-1.1,-2,1], [-2,2,1], [-1.5,4,1], [0,5,1], [1,3,1], [1.5,4,1],
                         [2,-3,-1], [-2,-3,-1], [2,1,-1], [3.1,3,-1], [3,5,-1], [1,6,-1], [-2,5,-1], [-3,3,-1]])
        self.data_size = len(data)
        return data
        
    def separable_2d(self, seed, n_points, vect_w=None):
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
 
        return data
        
    def separable_2d_circle(self, seed, n_points, vect_w=None):
        np.random.seed(seed)
        
        if vect_w == None:
            vect_w = [1.5, 0.3, 1]
        else:
            vect_w = vect_w

        dim_x = 2
        data_dim = dim_x + 1 # leading 1 and class value
        data = np.ones(data_dim * n_points).reshape(n_points, data_dim)
        
        x_r = vect_w[0] + vect_w[2] + 0.1
        x_l = vect_w[0] - vect_w[2] - 0.1
        x_range = x_r - x_l
        x_middle = (x_range/2) - vect_w[0]
        
        y_r = vect_w[1] + vect_w[2] + 0.1
        y_l = vect_w[1] - vect_w[2] - 0.1
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
 
        return data
        