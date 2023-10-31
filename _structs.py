# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:54:50 2023

@author: hhuang91
"""

class gStruct(object):
    def __init__(self):
        self.PixSize = []
        self.XYdim = []
        self.SDD = 0
        self.SAD = 0
        self.angleNum = 0
        self.angle = []
        self.u0 = 0
        self.v0 = 0
        self.pm = []
        self.pm_set = 0
    def setFromSioLoaded(self, g_str):
        self.PixSize = g_str['PixSize'][0][0][0]
        self.XYdim = g_str['UVWdim'][0][0][0]
        self.SDD = g_str['SDD'][0][0][0][0]
        self.SAD = g_str['SAD'][0][0][0][0]
        self.angle = g_str['angle'][0][0][0]
        self.angleNum = len(self.angle)
        self.u0 = g_str['u0'][0][0][0][0]
        self.v0 = g_str['v0'][0][0][0][0]
        tmp = g_str['pm'][0][0]
        self.pm = tmp
        self.pm_set = 1