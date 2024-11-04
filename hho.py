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
# An implementation of the Harris' Hawks Optimization algorithm
# adapted for feature selection. The original algorithm code is
# directly ported from the Matlab code in Ali Asghar Heidari's
# official repository
#
# https://github.com/aliasgharheidaricom/Harris-Hawks-Optimization-Algorithm-and-Applications/tree/master
#
# Certain comments were copied "as is", along with their grammar errors.
#
#
# The main paper is
# Harris hawks optimization: Algorithm and applications
# Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
# Future Generation Computer Systems,
# DOI: https://doi.org/10.1016/j.future.2019.02.028
#
# The feature selection adaptation is from
# A model for Mesothelioma cancer diagnosis based on feature selection using Harris hawk optimization algorithm
# Farehe Zadsafar, Hamed Tabrizchi, Sepideh Parvizpour, Jafar Razmara, Shahriar Lotfi
# Computer Methods and Programs in Biomedicine Update
# https://doi.org/10.1016/j.cmpbup.2022.100078
#

import numpy as np
import random
import sklearn
from numpy.random import random_sample as rand
from math import gamma as gamma

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC

# The sigmoid function, used to binarize
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# The Levy function, as implemented in the Matlab code.
def Levy(dim):
    beta = 1.5
    sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) /
             (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim)*sigma
    v = np.random.randn(1, dim)
    return u/np.abs(v)**(1/beta)


class HHO:
    def __init__(self, obj_func, N, T, lb, ub, dim):
        self.obj_func = obj_func
        self.N = N                          # Nuber of hawks
        self.T = T                          # number of iterations
        self.ub = np.array([ub]).flatten()  # upper bound for feature values
        self.lb = np.array([lb]).flatten()  # lower bound for feature values
        self.dim = dim                      # dimension of data vectors, also
        # is the number of features.
        # The flock of hawks (solution candidates)
        self.hawks = np.zeros((N, dim))

        self.initialize_hawks()

    def initialize_hawks(self):

        # If the upper (and lower) bound is a scalar, take it as a common limit
        # to all components(=features), generate N random vectors (hawks).
        if self.ub.shape[0] == 1:
            self.hawks = rand((self.N, self.dim)) * \
                (self.ub - self.lb) + self.lb
        # Otherwise the bounds are vectors of length dim, so each component has a
        # different value range. Form random hawks obeying these bounds.
        else:
            for i in range(0, dim):
                self.hawks[:, i] = rand(
                    (N, 1)) * (self.ub[i] - self.lb[i]) + self.lb[i]

    # Programming remark:
    # In the code below, due to the algebraic operations on the right sides of assignments,
    # self.hawks[i] is assigned a fresh (deep) copy object each time, so np.copy() is not needed.

    def optimize(self):

        # Initial dummy values for the rabbit and its energy
        # Rabbit_Energy stores the fitness (accuracy) of the best
        # solution obtained so far
        Rabbit_Location = np.zeros((1, self.dim))
        Rabbit_Energy = float('inf')

        # This will keep track of Rabbit_Energy (= accuracy)
        CNVG = np.zeros(self.T)

        for t in range(0, self.T):

            # Move the hawks back into the region boundaries if necessary
            self.hawks = np.maximum(np.minimum(self.hawks, self.ub), self.lb)

            # Find the most accurate hawk (the rabbit)
            for i in range(0, self.N):
                fitness = self.obj_func(self.hawks[i])
                if fitness < Rabbit_Energy:
                    Rabbit_Energy = fitness
                    Rabbit_Location = self.hawks[i].copy()

            # factor to show the decreaing energy of rabbit
            E1 = 2*(1 - t/self.T)

            # Update the location of Harris' hawks
            for i in range(0, self.N):
                E0 = 2*rand() - 1  # Choose a random number -1<E0<1
                # escaping energy of rabbit, a random
                Escaping_Energy = E1*(E0)
                # number in (-E1, E1)

                if abs(Escaping_Energy) >= 1:
                    # Exploration: Harris' hawks perch randomly
                    # based on 2 strategy:

                    q = rand()
                    rand_Hawk_index = np.random.randint(self.N)
                    hawk_rand = self.hawks[rand_Hawk_index]
                    if q >= 0.5:
                        # perch based on other family members
                        self.hawks[i] = hawk_rand - rand() * \
                            np.abs(hawk_rand - 2*rand() * self.hawks[i])

                    else:
                        # perch on a random tall tree (random
                        # site inside group's home range)
                        self.hawks[i] = (Rabbit_Location - np.mean(self.hawks, axis=0)) - \
                            rand()*((self.ub-self.lb)*rand() + self.lb)

                elif abs(Escaping_Energy) < 1:
                    # Exploitation:
                    # Attacking the rabbit using 4 strategies regarding
                    # the behavior of the rabbit

                    # phase 1: surprise pounce (seven kills)
                    # surprise pounce (seven kills): multiple, short rapid
                    # dives by different hawks

                    r = rand()  # probablity of each event

                    if r >= 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege
                        self.hawks[i] = Rabbit_Location - Escaping_Energy * \
                            abs(Rabbit_Location-self.hawks[i])

                    if r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege
                        # random jump strength of the rabbit
                        Jump_strength = 2*(1-rand())
                        self.hawks[i] = (Rabbit_Location - self.hawks[i])-Escaping_Energy*abs(
                            Jump_strength*Rabbit_Location-self.hawks[i])

                    # phase 2: performing team rapid dives (leapfrog movements)
                    # Soft besiege, rabbit try to escape by many zigzag deceptive motions
                    if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                        Jump_strength = 2*(1-rand())
                        hawk1 = Rabbit_Location - Escaping_Energy * \
                            abs(Jump_strength*Rabbit_Location-self.hawks[i])

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            # Perform levy-based short rapid dives
                            # around the rabbit
                            hawk2 = Rabbit_Location-Escaping_Energy * \
                                abs(Jump_strength*Rabbit_Location-self.hawks[i])+rand(self.dim)*Levy(self.dim)
                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

                    # Hard besiege, rabbit try to escape by many zigzag deceptive motions
                    if r < 0.5 and abs(Escaping_Energy) < 0.5:
                        # hawks try to decrease their average location with the rabbit
                        Jump_strength = 2*(1-rand())
                        hawk1 = Rabbit_Location-Escaping_Energy * \
                            abs(Jump_strength*Rabbit_Location -
                                np.mean(self.hawks, axis=0))

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            # Perform levy-based short rapid dives
                            # around the rabbit
                            hawk2 = Rabbit_Location-Escaping_Energy * \
                                abs(Jump_strength*Rabbit_Location-np.mean(self.hawks,
                                    axis=0))+rand(self.dim)*Levy(self.dim)
                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

            CNVG[t] = Rabbit_Energy

        return Rabbit_Energy, Rabbit_Location, CNVG

    # Some comments are omitted here since they are the same as above. The only difference
    # with the continuous version is that, the updated hawks are computed using the same
    # formulae as above and then the sigmoid values of their components are compared against
    # random values from (0,1) to binarize the results.
    #
    # Also, the upper/lower bounds are taken as 1/0.

    def optimize_binary(self):

        r_shape = (1, self.dim)
        Rabbit_Location = np.zeros(r_shape)
        Rabbit_Energy = float('inf')

        self.ub = 1
        self.lb = 0

        CNVG = np.zeros(self.T)

        # Binarize the randomly initialized hawks using the same formulae as in the optimization loops
        self.hawks = (sigmoid(np.maximum(np.minimum(
            self.hawks, self.ub), self.lb)) > rand(self.hawks.shape)) * 1

        for t in range(0, self.T):

            # Check boundaries
            #
            # Note [KIE]: In the binarized version, bounds check is unnecessary.
            #
            # self.hawks = np.maximum(np.minimum(self.hawks, self.ub), self.lb)

            for i in range(0, self.N):
                fitness = self.obj_func(self.hawks[i])
                if fitness < Rabbit_Energy:
                    Rabbit_Energy = fitness
                    # print(f"\nNew fitness {-fitness}")
                    # print(f"Features={self.hawks[i]}")
                    Rabbit_Location = self.hawks[i].copy()

            # factor to show the decreaing energy of rabbit
            E1 = 2*(1 - t/self.T)

            # Update the location of Harris' hawks
            for i in range(0, self.N):
                E0 = 2*rand() - 1  # -1<E0<1
                Escaping_Energy = E1*(E0)  # escaping energy of rabbit

                if abs(Escaping_Energy) >= 1:
                    # Exploration: Harris' hawks perch randomly
                    # based on 2 strategy:

                    q = rand()
                    rand_Hawk_index = np.random.randint(self.N)
                    hawk_rand = self.hawks[rand_Hawk_index]
                    if q < 0.5:
                        # perch based on other family members
                        self.hawks[i] = (sigmoid(
                            hawk_rand - rand() * np.abs(hawk_rand - 2*rand() * self.hawks[i])) > rand(r_shape)) * 1

                    else:
                        # perch on a random tall tree (random
                        # site inside group's home range)
                        self.hawks[i] = (sigmoid((Rabbit_Location - np.mean(self.hawks, axis=0))-rand()*(
                            (self.ub-self.lb)*rand() + self.lb)) > rand(r_shape)) * 1

                elif abs(Escaping_Energy) < 1:
                    # Exploitation:
                    # Attacking the rabbit using 4 strategies regarding
                    # the behavior of the rabbit

                    # phase 1: surprise pounce (seven kills)
                    # surprise pounce (seven kills): multiple, short rapid
                    # dives by different hawks

                    r = rand()  # probablity of each event

                    if r >= 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege
                        self.hawks[i] = (sigmoid(
                            Rabbit_Location - Escaping_Energy * abs(Rabbit_Location-self.hawks[i])) > rand(r_shape)) * 1

                    if r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege
                        # random jump strength of the rabbit
                        Jump_strength = 2*(1-rand())
                        self.hawks[i] = (sigmoid((Rabbit_Location - self.hawks[i])-Escaping_Energy*abs(
                            Jump_strength*Rabbit_Location-self.hawks[i])) > rand(r_shape)) * 1

                    # phase 2: performing team rapid dives (leapfrog movements)
                    # Soft besiege % rabbit try to escape by many zigzag deceptive motions
                    if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                        Jump_strength = 2*(1-rand())
                        hawk1 = (sigmoid(Rabbit_Location - Escaping_Energy*abs(
                            Jump_strength*Rabbit_Location-self.hawks[i])) > rand(r_shape)) * 1

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            # Perform levy-based short rapid dives
                            # around the rabbit
                            hawk2 = (sigmoid(Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-np.mean(
                                self.hawks, axis=0))+rand(r_shape)*Levy(self.dim)) > rand(r_shape))*1
                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

                    # Hard besiege % rabbit try to escape by many zigzag deceptive motions
                    if r < 0.5 and abs(Escaping_Energy) < 0.5:
                        # hawks try to decrease their average location with the rabbit
                        Jump_strength = 2*(1-rand())
                        hawk1 = (sigmoid(Rabbit_Location-Escaping_Energy*abs(Jump_strength *
                                 Rabbit_Location-np.mean(self.hawks, axis=0))) > rand(r_shape)) * 1

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            # Perform levy-based short rapid dives
                            # around the rabbit
                            hawk2 = (sigmoid(Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-np.mean(
                                self.hawks, axis=0))+rand(r_shape)*Levy(self.dim)) > rand(r_shape)) * 1
                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

            CNVG[t] = Rabbit_Energy

        return Rabbit_Energy, Rabbit_Location, CNVG
