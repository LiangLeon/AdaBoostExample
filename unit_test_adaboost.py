# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 15:00:26 2015

@author: LiangLeon
"""
import numpy as np
import unittest
import DataGen
import DecisionStumps

class TestDecisionStumpsMethods(unittest.TestCase):

  def test_default_data_1(self):
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data_1()
    weights = np.linspace(1, 1, len(data))
    weights /= len(data)
    MyDS = DecisionStumps.DecisionStumps()
    self.assertEqual([[0.30000000000000004, -1, 0, 0.5], [0.30000000000000004, -1, 1, -2.5]], MyDS.get_hypothesis(len(data), MyDataGen.data_dim, data, weights))

  def test_default_data_2(self):
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data_2()
    weights = np.linspace(1, 1, len(data))
    weights /= len(data)
    MyDS = DecisionStumps.DecisionStumps()
    self.assertEqual([[0.25, 1, 0, -0.5], [0.25, -1, 1, -1.75]], MyDS.get_hypothesis(len(data), MyDataGen.data_dim, data, weights))

  def test_default_data_3(self):
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data_3()
    weights = np.linspace(1, 1, len(data))
    weights /= len(data)
    MyDS = DecisionStumps.DecisionStumps()
    self.assertEqual([[0.25, -1, 0, 1.75], [0.375, 1, 1, -2.5]], MyDS.get_hypothesis(len(data), MyDataGen.data_dim, data, weights))
    
  def test_get_g_and_predect_result_default_data_3(self):
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data_3()
    weights = np.linspace(1, 1, len(data))
    weights /= len(data)
    MyDS = DecisionStumps.DecisionStumps()
    self.assertEqual([[1.75, 0, -1], [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1], 0.25], MyDS.get_g_and_predect_result(len(data), MyDataGen.data_dim, data, weights))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDecisionStumpsMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)