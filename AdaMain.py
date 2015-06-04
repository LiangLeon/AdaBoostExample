# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:02 2015

@author: leftword
"""

import DataGen
import AdaBoost

if __name__ == '__main__':
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data_1()
    Boost = AdaBoost.AdaBoost()
    Boost.get_adaboost_model(data, desired_error=0.01)
    g_s = Boost.g_stack
    a_s = Boost.alpha_stack
    error = Boost.total_error
    cur_round = Boost.current_round