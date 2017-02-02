import pandas as pd
import numpy  as np
import pylab  as pyl

import pydotplus 

from random import seed
from random import randrange
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from io import StringIO
from inspect import getmembers
import graphviz
from IPython.display import Image  

dataset = pd.read_csv('data_banknote_authentication.csv')
print(dataset.shape)
dat = dataset.values
print('reading dat: ',len(dat))

# Split a dataset into k folds
def cross_validation_split(data, n_folds):
    print('Entring cross_validation_split n_folds = ', n_folds)
    dat_split = list()
    dim = len(data)
    print('dim =', dim)
    dat_copy  = list(data)
    fold_size = int(dim / n_folds)
    print('fold_size =', fold_size)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dat_copy))
            #print('index = ', index)
            fold.append(dat_copy.pop(index))
        dat_split.append(fold)
    print('Exiting cross_validation_split n_folds = ')        
    return dat_split
   
#Split a dataset based on an attribute and an attribute value
def test_split(index, value, data):
    left, right = list(), list()
    for row in data:
        if row[index]< value:
            left.append(row)
        else:
            right.append(row)
    return left, right
#Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size ==0:
                continue
            proportion = [row[-1] for row in group].count(class_value)/float(size)
            gini += (proportion * (1.0 - proportion))
    return gini        
#Select the best split point for a dataset
def get_split(data):
    class_values = list(set(row[-1] for row in data))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(data[0])-1):
        for row in data:
            groups = test_split(index, row[index], data)
            gini   = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    print('b_index =', b_index)
    print('b_value =', b_value)
    print('b_score =', b_score)
    print('dim groups =', len(groups))
    print('dim right=', len(b_groups[0]))
    print('dim left=', len(b_groups[1]))
    return {'index':b_index, 'value': b_value, 'groups': b_groups}
#Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
#Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    #check for a no split
    if not left or not right:
        node['left']=node['right']= to_terminal(left+right)
        return
    #check for max depth
    if depth >= max_depth:
        node['left'], node['right']= to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left']= to_terminal(left)
    else:
        node['left']=get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right)<= min_size:
        node['right']= to_terminal(right)
    else:
        node['right']= get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
#Print tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))
#build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    print('1 length root = ',len(root))
    split(root, max_depth, min_size, 1)
    print('2 length root = ',len(root))
    print_tree(root)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
#Classification and  regression tree algorithm
def decision_tree(train, test, maxd, mins):
    print('decision_tree: maxd =', maxd)
    print('decision_tree: mins =', mins)
    tree_f = build_tree(train, maxd, mins)
    print('decision_tree:', len(train))

    atrain = np.array(train)
    atr    = atrain[:, 0:4]
    ay     = atrain[:, 4]
    print('decision_tree: dim(atr)=', atr.shape)
    print('decision_tree: dim(ay)= ', ay.shape)
    dt=tree.DecisionTreeClassifier(min_samples_split=mins,     random_state=1,max_depth=maxd)
    dt.fit(atr, ay)
    export_graphviz(dt, out_file='tree.dot')
    
    atest = np.array(test)
    print('decision_tree: dim(atest)=', atest.shape)
    x_test = atest[:, 0:4]
    result = dt.predict(x_test)
    predictions=list()
    for row in test:
        prediction = predict(tree_f, row)
        predictions.append(prediction)
    return(predictions, result)  
#Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct +=1
    return correct / float(len(actual)) * 100.0
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
    folds = cross_validation_split(data, n_folds)
    print(' evaluation_algorithm: number folds = ', len(folds))
    scores  = list()
    scores2 = list()
    for f in range(len(folds)):
        print('***********fold =',f+1,'****************')
        train_set = list(folds)
        fold      = folds[f]
        del train_set[f]
        train_set = sum(train_set, [])
        test_set  = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, result = algorithm(train_set, test_set, *args)
        actual    = [row[-1] for row in fold]
        accuracy  = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        accuracy  = accuracy_metric(actual, result)
        scores2.append(accuracy)
    print(' evaluation_algorithm: number folds = ', len(folds))    
    return scores, scores2
     
# Parameters of the method
#folds are parts in which the data set is divided in the study
#max_depth is length of the branches of the tree.
#min_size is the minimum size of a branch.
seed(1)
n_folds   = 5
max_depth = 4
min_size  = 10
scores, scores2 = evaluate_algorithm(dat,decision_tree,n_folds,max_depth,min_size)
print('Scores: %s' % scores)
print('Scores2: %s' % scores2)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print('Mean Accuracy2: %.3f%%' % (sum(scores2)/float(len(scores2))))