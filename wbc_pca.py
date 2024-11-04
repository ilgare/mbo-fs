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
# The Principal Component Analysis (PCA) performance
# on the Wisconsin Breast Cancer dataset:
#
# 
# https://doi.org/10.24432/C5DW2B
# https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# UCI Machine Learning Repository
# 
import pandas as pd
import numpy as np
import sklearn
import random
import platform
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA, KernelPCA

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# We select a random seed for the RNG.
# We will use it to create DecisionTreeClassifiers with identical seeds.
dt_seed = np.random.randint(424242)

# Names of the feature selection methods
names = ['pca', 'pca_poly', 'pca_rbf']

# List to store accuracies
accuracies = [0] * len(names)

# Number of features to be selected
num_feat = 13

num_trials = 25

for nt in range(0,num_trials):

    print(f"\nTrial {nt+1}/{num_trials}\n")
   
    # For cross-validation: We use a different random test set
    # at each trial.
    # Split data, 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalize data using the standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=17)
    pca.fit(X_train)
    X_train_t = pca.transform(X_train)
    X_test_t = pca.transform(X_test)
    
    model = DecisionTreeClassifier(random_state=dt_seed)
    model.fit(X_train_t, y_train)
    predictions = model.predict(X_test_t)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[0] += accuracy

    # PCA Polynomial kernel
    pca = KernelPCA(n_components=17, kernel='linear')
    pca.fit(X_train)
    X_train_t = pca.transform(X_train)
    X_test_t = pca.transform(X_test)
    
    model = DecisionTreeClassifier(random_state=dt_seed)
    model.fit(X_train_t, y_train)
    predictions = model.predict(X_test_t)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[1] += accuracy

    # PCA rbf kernel
    pca = KernelPCA(n_components=17, kernel='rbf')
    pca.fit(X_train)
    X_train_t = pca.transform(X_train)
    X_test_t = pca.transform(X_test)
    
    model = DecisionTreeClassifier(random_state=dt_seed)
    model.fit(X_train_t, y_train)
    predictions = model.predict(X_test_t)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[2] += accuracy



# After the trial loop is over, compute averages
for i in range(0,len(accuracies)):
    accuracies[i] /= num_trials
        
# Name of the output file, with a timestamp
now = datetime.now()
output_fname=f"output_wbc_pca_"+now.strftime("%d-%m-%y_%H:%M:%S")+".txt"

with open(output_fname, 'w') as f:

    # The table with the averages
    df_dict = {}
    df_dict['Selector name'] = names
    df_dict['Average accuracy'] = accuracies

    df = pd.DataFrame(df_dict)

    output_txt = df.to_markdown(index=False, tablefmt='plain', colalign=['center']*len(df.columns), floatfmt=".4f")

    f.write(output_txt)
    print(output_txt)

    f.write(f"\n\nNumber of trials = {num_trials}")
    f.write(f"\n\nNumber of features = {num_feat}")
   
    f.write(f"\n\nSystem:\n{platform.platform()}\n")
