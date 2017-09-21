# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:41:22 2017

@author: Awal
"""

import numpy as np
import matplotlib.pyplot as plt

import math

import pattern_utils
import population_search


def initial_population(region, scale = 10, pop_size=20):
    '''
    
    '''        
    # initial population: exploit info from region
    rmx, rMx, rmy, rMy = region
    W = np.concatenate( (
                 np.random.uniform(low=rmx,high=rMx, size=(pop_size,1)) ,
                 np.random.uniform(low=rmy,high=rMy, size=(pop_size,1)) ,
                 np.random.uniform(low=-np.pi,high=np.pi, size=(pop_size,1)) ,
                 np.ones((pop_size,1))*scale
                 #np.random.uniform(low=scale*0.9, high= scale*1.1, size=(pop_size,1))
                        ), axis=1)    
    return W

def test_particle_filter_search():
    if True:
        # use image 1
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(True)
        ipat = 2 # index of the pattern to target
    else:
        # use image 2
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(True)
        ipat = 0 # index of the pattern to target
        
    # Narrow the initial search region
    pat = pat_list[ipat] #  (100,30, np.pi/3,40),
    #    print(pat)
    xs, ys = pose_list[ipat][:2]
    region = (xs-20, xs+20, ys-20, ys+20)
    scale = pose_list[ipat][3]
    print (scale)
        
    pop_size=60
    W = initial_population(region, scale, pop_size)
    print(W)
    print(W.shape)

#A = np.concatenate( (
#                 np.random.uniform(low=0,high=100, size=(20,1)) ,
#                 np.random.uniform(low=0,high=100, size=(20,1)) ,
#                 np.random.uniform(low=-np.pi,high=np.pi, size=(20,1)) ,
#                 np.ones((20,1))*40
#                 #np.random.uniform(low=scale*0.9, high= scale*1.1, size=(pop_size,1))
#                        ), axis=1)
#for pose in A:
#    print(pose)       
#
#test_particle_filter_search()
#
#class Person(object):
#    def __init__(self,name):
#        self.name = name
#    
#    def printStuff(self):
#        return self.name
#
#class Gary(Person):
#    def __init__(self):
#        name = "Gary"
#        super().__init__(name)
#        
#G = Gary()
#A = G.printStuff()
#B = super(Gary,G).printStuff()
#print(A)
#print(B)
#
#imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(True)
#ipat = 2 # index of the pattern to target
#
#pat = pat_list[ipat]
#
#C,Vp = pat.evaluate(imd,(100,30, np.pi/3,40))
#
#print(C)
#print(Vp)
#
#W = np.array([1,2,3])
#Y = np.array([4])
#W = np.append(W,Y)
#print(W)

mutations = np.concatenate((
        np.random.choice([-1,0,1], 3, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1),
        np.random.choice([-1,0,1], 3, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1),
        np.random.choice([-0.0174533,0,0.0174533], 3, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1),  #0.0174533 is approximatly 1 degree in radians
        np.random.choice([-1,0,1], 3, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1)
            ), axis=1)
        

print (mutations)

