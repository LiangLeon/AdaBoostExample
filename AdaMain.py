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
    g_s, a_s = Boost.get_adaboost_model(data, desired_error=0.01)
    
    predict_answer = Boost.predict_result(g_s,a_s,data[1])
    print data[1,-1] == predict_answer