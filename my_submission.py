'''

2017 IFN680 Assignment

Instructions: 
    - You should implement the class PatternPosePopulation
    
Authors: Awal Singh, Ray Bastien, Frank Pham
Date last modified: 24/9/2017
Python Version: 3.6

How to recreate the experimental results:
    Simply running this file will recreate the experimental results used in the report for image 1. However, this takes a very 
    long time. Hence we would suggest looking at the "RunTestsForEachCombination" function and changing the testsPerCombination
    variable when running this function to reduce the amount of tests that will be done. To recreate the tests for image 2 simply 
    change the True to a False in the "if" statement in the "ParticleFilterSearch_ExperimentalTests" function. All of the code
    from the original "my_submission.py" file that was provided still remains the only changes were completing the evaluate and
    mutate and sections and adding two more functions to complete the experimental tests. 

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
        These 3 lines of code ensure that the position of the pattern remains inside the boundary of the distance image.
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
    
def ParticleFilterSearch_ExperimentalTests(pop_size, iterations): #
    '''
    We created this function in order to be able to test the particle filter search and find the experimental results.
    This function is very similar to the "test_particle_filter_search" function with minor changes. These include having the 
    population size and number of iterations as formal parameters, commenting out the uncessary code and returning the 
    pop.best_cost and pop.best_w variables.
    
    @param
        pop_size: The initial population size for the particle filter search
        iterations: The number of iterations run in the particle filter search
    
    @return
        pop.best_cost: The lowest cost found in the particle filter search
        pop.best_w: The lowest cost pose found in the particle filter search
    
    '''
    
    if True:
        # use image 1
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(False)
        ipat = 2 # index of the pattern to target
    else:
        # use image 2
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(False)
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
    
    #plt.plot(Lc)
    #plt.title('Cost vs generation index')
    #plt.show()
    
    return (pop.best_cost, pop.best_w)
    
        
    #pattern_utils.display_solution(pat_list, 
    #                  pose_list, 
    #                  pat,
    #                  pop.best_w)
                      
    #pattern_utils.replay_search(pat_list, 
    #                  pose_list, 
    #                  pat,
    #                  Lw)
    
def RunTestsForEachCombination(testsPerCombination = 100):
    '''
    This was the code used to find the experimental results for the assignment. Here we are running 100 tests for every possible
    combination of initial population size and number of iterations. The variable "combinations" contains every two integers that
    multiply to 10000 which was our chosen computational budget. We then run the "ParticleFilterSearch_ExperimentalTests" function
    100 times on each of these combinations and stored the best pose and its cost in a text file called "experiment.txt" for 
    each test. We also find the mean best cost for a given combination over the 100 tests, and create a box plot to display the
    costs in each test so we could visualise the varience in the data.
    
    Note: This function takes a long time to run and so it might be worth reducing the number of tests per combination or the total
    number of combinations tested. We also found that some of the combinations with a particularly high number of iterations did
    cause errors to occur when they were run. We suspect this has to do with the size of the pose shrinking each iteration until 
    it became a degenerate point. Fortunatly this simply means that these combinations are not worth testing since the best pose
    they find are not very close to the correct pose they should find. 
    
    @param
        testsPerCombination: The number of tests run for a given combination
    
    '''    

    combinations = [
                   [10000,1],[5000,2],[2500,4],[2000,5],[1250,8],
                   [1000,10],[625,16],[500,20],[400,25],[250,40], 
                   [200,50],[125,80],[100,100],[80,125],[50,200],
                   [40,250],[25,400],[20,500],[16,625],[10,1000],
#                   [8,1250],[5,2000],[4,2500],[2,5000],[1,10000]  # These combinations often don't work.
                    ]
    bestCosts = []
    myFile = open('experiment.txt','w')       
    for comb in combinations:
        popsize = comb[0]
        iteration = comb[1]
        bestCosts = []
        bestPoses = []         
        for i in range(testsPerCombination):
            curCost , curPose = ParticleFilterSearch_ExperimentalTests(popsize,iteration)
            bestCosts.append(curCost)
            bestPoses.append(curPose)
            print(str(i+1) + '-' + str(curCost))  #This is used for testing that the code is working.
    
        mean = np.mean(bestCosts)
        myFile.write("Pop: " + str(popsize) + " - Iter: " + str(iteration) + "\n")
        myFile.write("Mean: " + str(mean) + "\n")        
        for i in range(len(bestCosts)):
            myFile.write("C:" + str(bestCosts[i]) + " P:" + str(bestPoses[i]) + "\n")
        myFile.write("\n") 
        bestCosts = np.reshape(bestCosts,(-1,1))            
        plt.boxplot(bestCosts)
        plt.title('Best Costs | Pop: ' + str(popsize) + ' - Iter: ' + str(iteration) )        
        plt.show()                      
        print(mean)
    myFile.close()
#------------------------------------------------------------------------------        

if __name__=='__main__':
    #test_particle_filter_search()
    RunTestsForEachCombination()

    
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
    
