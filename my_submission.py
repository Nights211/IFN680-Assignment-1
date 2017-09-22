'''

2017 IFN680 Assignment

Instructions: 
    - You should implement the class PatternPosePopulation

'''

import numpy as np
import matplotlib.pyplot as plt


import pattern_utils
import population_search

#------------------------------------------------------------------------------

class PatternPosePopulation(population_search.Population):
    '''
    
    '''
    def __init__(self, W, pat):
        '''
        Constructor. Simply pass the initial population to the parent
        class constructor.
        @param
          W : initial population
        '''
        self.pat = pat
        super().__init__(W)
    
    def evaluate(self):
        '''
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w 
            self.best_cost 
        @return 
           best cost of this generation            
        '''
 
        	# INSERT YOUR CODE HERE
        '''
        These 3 lines of code ensure that the position of the pattern remains inside the boundary of the distance image funciton.
        '''
        height, width = self.distance_image.shape[:2]       
        np.clip(self.W[:,0],0,width-1,self.W[:,0])
        np.clip(self.W[:,1],0,height-1,self.W[:,1])
        
        '''
        Here we are creating a cost array "self.C" and evaluating the cost of individual pose vectors "W[i,:]" by calling the evaluate 
        function in the pattern_utils file. This evaluate function calulates the cost of a given pose vector by finding the average 
        distance between a pixel on the indivual (The pattern created by the pose vector) and the edge pixel (The shapes in imf).
        We store the cost of a pose vector W[i,:] into self.C[i]. 
        '''
        self.C = np.array([])
        for pose in self.W:
            score, Vp = self.pat.evaluate(self.distance_image,pose)
            self.C = np.append(self.C, np.array([score]))
            
        '''
        We now want to find the best cost of this generation "cost_min" and the best cost overall "self.best_cost" and the 
        pose vector which has the best cost overall "self.best_w". To do this we find the smallest value in self.C and check 
        if it is smaller than the best cost of previous generations and update the values if necessary. 
        '''
        i_min = self.C.argmin()
        cost_min = self.C[i_min]
        if cost_min<self.best_cost:
            self.best_w = self.W[i_min].copy()
            self.best_cost = cost_min
        return cost_min


    def mutate(self):
        '''
        Mutate each individual.
        The x and y coords should be mutated by adding with equal probability 
        -1, 0 or +1. That is, with probability 1/3 x is unchanged, with probability
        1/3 it is decremented by 1 and with the same probability it is 
        incremented by 1.
        The angle should be mutated by adding the equivalent of 1 degree in radians.
        The mutation for the scale coefficient is the same as for the x and y coords.
        @post:
          self.W has been mutated.
        '''
        
        assert self.W.shape==(self.n,4)

        	# INSERT YOUR CODE HERE        
        
        '''
        mutations is a self.n by 4 matrix that can randomly have the values [-1,0,1] on the first, second and fourth coloums.
        On the thrid coloum it can randomly have the values [-0.0174533,0,0.0174533] where 0.0174533 is equal to one degree 
        in radians. The probability of it choosing any these values is 1/3 and so it has a uniform random distribution.
        We then add the mutations matrix to the self.W matrix in order to mutate the valiables for the particle filter search.
        
        '''
        mutations = np.concatenate((
        np.random.choice([-1,0,1], self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1),
        np.random.choice([-1,0,1], self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1),
        np.random.choice([-0.0174533,0,0.0174533], self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1),  #0.0174533 is approximatly 1 degree in radians
        np.random.choice([-1,0,1], self.n, replace=True, p = [1/3,1/3,1/3]).reshape(-1,1)
            ), axis=1)
        
        self.W = self.W+mutations
    
    def set_distance_image(self, distance_image):
        self.distance_image = distance_image

#------------------------------------------------------------------------------        

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

#------------------------------------------------------------------------------        
def test_particle_filter_search():
    '''
    Run the particle filter search on test image 1 or image 2of the pattern_utils module
    
    '''
    
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
        
    pop_size = 50
    W = initial_population(region, scale , pop_size)
    
    pop = PatternPosePopulation(W, pat)
    pop.set_distance_image(imd)
    
    pop.temperature = 5
    
    Lw, Lc = pop.particle_filter_search(40,log=True)
    
    plt.plot(Lc)
    plt.title('Cost vs generation index')
    plt.show()
    
    print(pop.best_w)
    print(pop.best_cost)
    
        
    pattern_utils.display_solution(pat_list, 
                      pose_list, 
                      pat,
                      pop.best_w)
                      
    pattern_utils.replay_search(pat_list, 
                      pose_list, 
                      pat,
                      Lw)      
#------------------------------------------------------------------------------    
    
def ParticleFilterSearch_ExperimentalTests(pop_size = 50, iterations = 40): #Changed this line so that when we call the function we can also specify the popsize and interations 
    '''
    We created this function in order to be able to test the particle filter seach and find the experimental results.
    This function is very similar to the test_particle_filter_search function with minor changes to make testing simpler.
    
    '''
    
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
        
    
    W = initial_population(region, scale , pop_size)
    
    pop = PatternPosePopulation(W, pat)
    pop.set_distance_image(imd)
    
    pop.temperature = 5
    
    Lw, Lc = pop.particle_filter_search(iterations,log=True)
    
    plt.plot(Lc)
    plt.title('Cost vs generation index')
    plt.show()
    
    print(pop.best_w)
    print(pop.best_cost)
    
        
    pattern_utils.display_solution(pat_list, 
                      pose_list, 
                      pat,
                      pop.best_w)
                      
    pattern_utils.replay_search(pat_list, 
                      pose_list, 
                      pat,
                      Lw)
#------------------------------------------------------------------------------        

if __name__=='__main__':
    ParticleFilterSearch_ExperimentalTests(1000,3)
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
#        
#    def test_2():
#        '''
#        Run the particle filter search on test image 2 of the pattern_utils module
#        
#        '''
#        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(False)
#        pat = pat_list[0]
#        
#        #region = (100,150,40,60)
#        xs, ys = pose_list[0][:2]
#        region = (xs-20, xs+20, ys-20, ys+20)
#        
#        W = initial_population_2(region, scale = 30, pop_size=40)
#        
#        pop = PatternPosePopulation(W, pat)
#        pop.set_distance_image(imd)
#        
#        pop.temperature = 5
#        
#        Lw, Lc = pop.particle_filter_search(40,log=True)
#        
#        plt.plot(Lc)
#        plt.title('Cost vs generation index')
#        plt.show()
#        
#        print(pop.best_w)
#        print(pop.best_cost)
#        
#        
#        
#        pattern_utils.display_solution(pat_list, 
#                          pose_list, 
#                          pat,
#                          pop.best_w)
#                          
#        pattern_utils.replay_search(pat_list, 
#                          pose_list, 
#                          pat,
#                          Lw)
#    
#    #------------------------------------------------------------------------------        
#        
    