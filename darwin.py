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
# The Migrating Birds Optimization (MBO) and Harris Hawk Optimization (HHO)
# feature selection performance on the DARWIN dataset:
#
# https://archive.ics.uci.edu/dataset/732/darwin
# UCI Machine Learning Repository
#  

import pandas as pd
import numpy as np
import sklearn
import random
import mbo
import hho
import time
import platform
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load dataset
X = pd.read_csv('darwin/X.csv')
X = X.iloc[:,1:]
y = pd.read_csv('darwin/y.csv')

# Convert P/H to 1/0
y['class'] = (y['class'] == 'P').astype(int)

# Combine and shuffle
X['class'] = y['class']
X = X.sample(frac=1)

# Split the targets again
y = X['class']
X = X.drop(columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize data using the standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Number of features
num_features = X_train.shape[1]

# We select a random seed for the RNG.
# We will use it to create DecisionTreeClassifiers with identical seeds.
dt_seed = np.random.randint(424242)

# Objective function to minimize (negative accuracy score)
def objective(bird):

    # Get the indices of the selected features 
    # based on the binary mask (feature filter)
    selected_features = np.where(bird==1)
    selected_features = selected_features[0]
    
    if len(selected_features) == 0:
        return float('-inf')  # Penalize empty feature set

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    model = DecisionTreeClassifier(random_state=dt_seed)
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, predictions)
    return -accuracy 

# Compare results of various classifiers using both the trimmed and full feature lists.
classifiers = [ GaussianNB(), DecisionTreeClassifier(random_state=dt_seed), SVC(kernel='linear'), LogisticRegression(solver='liblinear'), XGBClassifier()]

num_classifiers = len(classifiers)

# Names of the feature selection methods
names = ['full', 'mbo', 'hho']

# Dictionaries to store accuracy, execution time, number of features
# and selected features corresponding to the methods above.
accuracies = {}
exec_times = {}
num_feat = {}
selected_feat = {}

# Full set of fetures
selected_feat['full'] = list(range(0,num_features))


for nm in names:
    accuracies[nm] = [0] * num_classifiers
    exec_times[nm] = 0
    num_feat[nm] = 0


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

    # First part: We first make feature selections using MBO,HHO etc.
    # and record the statistics.

    # Initialize and run MBO
    
    mbo_max_iter = 20
    mbo_mutation_size = 45
    mbo_wing_length = 7
    mbo_num_tours = 4
    mbo_k = 5
    mbo_x = 2

    mbo_obj = mbo.MigratingBirdsOptimization(num_features=num_features,obj_func=objective,wing_length=mbo_wing_length, max_iter=mbo_max_iter, num_tours = mbo_num_tours, k=mbo_k, x=mbo_x, mutation_size=mbo_mutation_size)

    time1 = time.process_time()
    best_bird = mbo_obj.optimize()
    time2 = time.process_time()

    exec_times['mbo'] += time2 - time1

    # Get the list of indices of the selected features
    selected_features = np.where(best_bird[0]==1)
    selected_features = selected_features[0]
    num_feat['mbo'] += len(selected_features)
    selected_feat['mbo'] = selected_features

    print(f"\nMBO Accuracy={-best_bird[1]}\nMBO selected Features={best_bird[0]}\nFeatures indices: {selected_features}\n")
    ####### End of MBO block


    # Initialize and run HHO
    
    hho_N = 15
    hho_T = 90
    
    hho_obj = hho.HHO(obj_func=objective, N=hho_N, T=hho_T, lb=0, ub=1, dim=num_features)

    time1 = time.process_time()
    r_energy, r_location, CNVG = hho_obj.optimize_binary()
    time2 = time.process_time()

    exec_times['hho'] += time2 - time1

    # Get the list of indices of the selected features
    selected_features = np.where(r_location == 1)
    selected_features = selected_features[0]
    num_feat['hho'] += len(selected_features)
    selected_feat['hho'] = selected_features

    print(f"\nHHO Accuracy={-r_energy}\nHHO selected features={r_location}\nFeature indices: {selected_features}\n")
    ####### End of HHO block


    # Second part: Now we make predictions with various classifiers
    # using the features selected above by MBO, HHO, etc.
    for i in range(0,num_classifiers):

        for nm in names:
            clf = sklearn.base.clone(classifiers[i])
            clf.fit(X_train[:, selected_feat[nm]], y_train)
            y_pred = clf.predict(X_test[:, selected_feat[nm]])
            accuracy_sc = accuracy_score(y_test, y_pred)
            accuracies[nm][i] += accuracy_sc

# After the trial loop is over, compute averages
for nm in names:
    for i in range(0,num_classifiers):
        accuracies[nm][i] /= num_trials
        
    exec_times[nm] /= num_trials
    num_feat[nm] /= num_trials

# Name of the output file, with a timestamp
now = datetime.now()
output_fname=f"output_darwin_"+now.strftime("%d-%m-%y_%H:%M:%S")+".txt"

with open(output_fname, 'w') as f:

    # The table with the averages
    df_dict = {}

    clf_names = [ type(c).__name__ for c in classifiers]

    df_dict['Classifier name'] = clf_names

    for nm in names:
        df_dict[nm] = accuracies[nm]

    df = pd.DataFrame(df_dict)

    output_txt = df.to_markdown(index=False, tablefmt='plain', colalign=['center']*len(df.columns), floatfmt=".4f")

    f.write(output_txt)
    print(output_txt)



    f.write(f"\n\nNumber of trials = {num_trials}")

    f.write(f"\n\nMBO parameters:\nTotal birds={2*mbo_wing_length+1}, iterations={mbo_max_iter}, tours={mbo_num_tours},\nmutation size={mbo_mutation_size}, k={mbo_k}, x={mbo_x}\n")
    f.write(f"\n\nHHO parameters:\nT={hho_T},  N={hho_N}\n")

    df_dict = {}
    df_dict = {'Algorithm': names[1:], 'Avg exec time (s)':[exec_times[nm] for nm in names[1:]], 'Avg number of features':[num_feat[nm] for nm in names[1:]]}
    df = pd.DataFrame(df_dict)

    output_txt = df.to_markdown(index=False, tablefmt='plain', colalign=['center']*len(df.columns), floatfmt=".4f")

    f.write(output_txt)

    print(output_txt)
    
    f.write(f"\n\nSystem:\n{platform.platform()}\n")
    
    

    
