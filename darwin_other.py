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
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif

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

# We select a random seed for the RNG.
# We will use it to create DecisionTreeClassifiers with identical seeds.
dt_seed = np.random.randint(424242)

# Objective function to minimize (negative accuracy score)
def objective(mask):

    # Get the indices of the selected features 
    # based on the binary mask (feature filter)
    selected_features = np.where(mask==True)
    selected_features = selected_features[0]
    
    if len(selected_features) == 0:
        return float('-inf')  # Penalize empty feature set

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    model = DecisionTreeClassifier(random_state=dt_seed)
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy 


# Names of the feature selection methods
names = ['rfecv', 'model', 'kbest']

# List to store accuracies
accuracies = [0] * len(names)

# Number of features to be selected
num_feat = 224

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

    # RFECV
    estimator =  DecisionTreeClassifier()
    selector = RFECV(estimator, step=1, min_features_to_select=5)
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    accuracies[0] += objective(mask)

    # SelectFromModel
    estimator =  DecisionTreeClassifier()
    selector = SelectFromModel(estimator, threshold=-np.inf, max_features=num_feat)
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    accuracies[1] += objective(mask)

    # SelectKBest
    selector = SelectKBest(f_classif, k=num_feat)
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    accuracies[2] += objective(mask)

        

# After the trial loop is over, compute averages
for i in range(0,len(accuracies)):
    accuracies[i] /= num_trials
        
# Name of the output file, with a timestamp
now = datetime.now()
output_fname=f"output_darwin_others_"+now.strftime("%d-%m-%y_%H:%M:%S")+".txt"

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

