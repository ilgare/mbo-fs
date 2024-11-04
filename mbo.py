# Code for the paper
#
# The performance of Migrating Birds Optimization as a feature selection tool
#
# by
#
# Kemal Ilgar Eroglu, Elif Ercelik, Hatice Coban Eroglu
#
# 10/23/2024
#
#
# An implementation of the Migrating Birds Optimization algorithm
# adapted for feature selection.
#
#

import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC

# We select a random seed for the RNG.
# We will use it to create DecisionTreeClassifiers with identical seeds.
dt_seed = np.random.randint(424242)

class MigratingBirdsOptimization:
    def __init__(self, num_features, obj_func,  wing_length=10, max_iter=25, num_tours=4, k=5, x=2, mutation_size=2):
        self.obj_func = obj_func
        self.num_features = num_features
        self.wing_length = wing_length  # Number of birds in each wing, excluding the leader
                                        # If wing_length is l, there are 2*l+1 birds.
        self.max_iter = max_iter        # Maximum number of iterations. The leader is relocated
                                        # at the end of each iteration.
        self.num_tours = num_tours      # Number of "tours" in each iteration.
        self.k = k                      # Number of neighbours to be considered when searching
                                        # for improvements.

        self.x = x                      # Number of neigbours shared with the bird behind.

        # Maximal number of parameters to be flipped (to select/remove features)
        # that occur in the generated neighbours.
        self.mutation_size = mutation_size

        # Birds are (feature_filter, fitness) pairs.
        # Fitness is simply the accuracy score.
        # Left/right_birds form the left/right wings of the V shape
        self.leader, self.left_birds, self.right_birds = self.initialize_birds()

    def initialize_birds(self):
        
        if self.k <= self.x:
            # The leader's neighbourhood should contain at least 2x+1
            # bird, the best one replaces the leader, then next best x birds
            # go the left wing, the next best x go to the right.
            print("k must be at least 2x+1")
            exit()


        # We cant modify more number of parameters than we have
        if self.mutation_size > self.num_features:
            print("Mutation size can not be larger than number of features")
            exit()
        
        l = self.wing_length
        
        # Generate random feature filters for the entire flock
        all_filters = np.random.randint(2,size=(2*l+1,self.num_features))

        allbirds = []

        # Generate the birds (feature list, fitness) pairs
        for feat in all_filters:
            fitness = self.obj_func(feat)
            allbirds += [(feat,fitness)]

        # Return the leader, left birds, right birds.
        # Leader is chosen randomly, not as the best-fitting one.
        return allbirds[0], allbirds[1:l+1], allbirds[l+1:2*l+1]

    def optimize(self):

        # Just some random assignment of "best result so far".
        # Use the initial leader's values as best, it will get updated
        # within the loop.
        best_feat = np.copy(self.leader[0])
        BF = self.leader[1]

        # Best fitnesses of 1 and 2 steps before.
        BF1 = BF
        BF2 = BF

        # new_BF will store the best score encountered so far in the current iteration
        new_BF = BF

        for count in range(0,self.max_iter):

            # Best fitness of the current and the previous two rounds 
            BF2 = BF1
            BF1 = BF
            BF = new_BF
               
            # print(f"\ncount={count} BF={-BF}\nFeatures={best_feat}")

            # The original MBO paper lists getting no improvement in 3 successive
            # iterations as a terminating condition. I opted not to implement this,
            # here the iteration limit determines the termination.
            #if (count >2) and (BF2 == BF):
            #    break

            for _ in range(0, self.num_tours):

                # Check the leader first, create k neighbours. Include
                # the current bird in the neighbours.
                nbs = [self.leader] + self.create_neigbours(self.leader, self.k)
                
                # Sort descending according to the fitness component
                nbs = sorted(nbs,key=lambda x:x[1])
                
                # The new leader is the best performing bird of the
                # neighbourhood.
                lead = nbs[0]

                # If this improves the best fitness known so far,
                # record it.
                if lead[1] < new_BF:
                    best_feat = lead[0]
                    new_BF = lead[1]

                self.leader = lead

                # The nighbours to be passed to the left and right wings
                left_share = nbs[1:1+self.x]
                right_share = nbs[1+self.x:1+2*self.x]

                kx = self.k - self.x

                # Run through the left wing
                for i in range(0,self.wing_length):
                    b = self.left_birds[i]
                    
                    # Create k-x random neighbours, x will be inherited
                    # from the front. Sort them by their accuracy score.
                    nbs = self.create_neigbours(b, kx)
                    nbs = [b] + nbs + left_share
                    nbs = sorted(nbs,key=lambda x:x[1])

                    # Select the best performing one
                    lead = nbs[0]
                    self.left_birds[i] = lead

                    # Update if the best score so far is improved
                    if lead[1] < new_BF:
                        best_feat = lead[0]
                        new_BF = lead[1]

                    # Next x best is shared with the bird behind
                    left_share = nbs[1:1+self.x]

                # Now the right wing, same as above
                for i in range(0,self.wing_length):
                    b = self.right_birds[i]
                    nbs = self.create_neigbours(b, kx)
                    nbs = nbs + right_share
                    nbs = sorted(nbs,key=lambda x:x[1])
                    lead = nbs[0]
                    self.right_birds[i] = lead

                    if lead[1] < new_BF:
                        best_feat = lead[0]
                        new_BF = lead[1]

                    right_share = nbs[1:1+self.x]

            # The tours are complete, relocate the leader to the back. In even 
            # iteration counts use the left wing, in the odd ones use the right.

            # Here we take a (deep) copy of the leader np.array, otherwise
            # it will get overwritten.
            temp = (np.copy(self.leader[0]),self.leader[1])
            
            if count%2 == 0:
                self.leader = self.left_birds[0]
                self.left_birds[:self.wing_length-1] = self.left_birds[1:self.wing_length]
                self.left_birds[self.wing_length-1] = temp
            else:
                self.leader = self.right_birds[0]
                self.right_birds[:self.wing_length-1] = self.right_birds[1:self.wing_length]
                self.right_birds[self.wing_length-1] = temp

        return (best_feat, BF)

    # bird: The bird to whom we create neighbours 
    # num_neigh: Number of neighbours to be created
    def create_neigbours(self, bird, num_neigh):
        neighs = []

        for i in range(0,num_neigh):

            # Select a random number between 1 and mutation_size, then select that many
            # random features from the feature list.
            to_be_flipped = np.random.randint(0,self.num_features,np.random.randint(1,self.mutation_size))

            # We take a fresh deep copy to get a disctinct fresh object
            n_ff = np.copy(bird[0])
            # Flip the selected elements of the feature filter
            n_ff[to_be_flipped]= 1 - n_ff[to_be_flipped]
            fitness = self.obj_func(n_ff)
            neighs += [(n_ff,fitness)]

        return neighs        
