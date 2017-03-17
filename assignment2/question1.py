import h5py
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from collections import OrderedDict
from operator import itemgetter
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.utils import np_utils
from sklearn.metrics import average_precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

import time

# Loading data 

dataD = scipy.io.loadmat('./DataD.mat')

features = dataD['fea']
labels = dataD['gnd']
labels[labels == -1] = 0
labels = labels.ravel()

print('Succesfully loaded data!')
print("Data D features shape:", features.shape)
print("Labels shape:", labels.shape)

# Z-score normalization

features_scaled = scale(features)
half = int(len(features_scaled) / 2)

X_train = features_scaled[:half]
X_test = features_scaled[half:]

y_train = labels[:half]
y_test = labels[half:]

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# k-NN

kf = KFold(n_splits=5)
k_accuracies = {}

k_neighbors = [i for i in range(1, 32, 2)]
for k in k_neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    acc = []
    for train, test in kf.split(X_train):
        clf.fit(X_train[train], y_train[train])
        acc.append(clf.score(X_train[test], y_train[test]))

    accuracy = np.mean(acc)
    k_accuracies[k] = accuracy

x = list(k_accuracies.values())
y = list(k_accuracies.keys())

plt.plot(y, x)
plt.title('k-NN Accuracy with different values of k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# SVM, grid-search for best C and sigma/gamma values

sigmas = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [1/(2 * sigma**2) for sigma in sigmas]

parameters = {'kernel':['rbf'], 'C':[0.1, 0.5, 1, 2, 5,10, 20, 50], 'gamma': gamma}
svr = SVC()
clf = GridSearchCV(svr, parameters, cv=5)
clf.fit(X_train, y_train)
print("Best accuracy:", clf.best_score_,  clf.best_params_)

svm = SVC(kernel='rbf', C=2, gamma=0.02)
svm.fit(X_train, y_train)
y_score = svm.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for SVM with best parameters')
plt.legend(loc="lower right")
plt.show()

# k-NN and SVM with the best parameters
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)

svm = SVC(kernel='rbf', C=2, gamma=0.02)
svm.fit(X_train, y_train)
svm_acc = svm.score(X_test, y_test)

print("KNN test accuracy:", knn_acc)
print("SVM test accuracy", svm_acc)

# Decision tree

clf = DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=1, max_leaf_nodes=(len(X_train) - 1))

clf.fit(X_train, y_train)
dt_acc = clf.score(X_test, y_test)

print("Decision tree test accuracy:", dt_acc)

# Random forest

randomforest = RandomForestClassifier(min_samples_split=10, min_samples_leaf=1, max_leaf_nodes=(len(X_train) - 1))

randomforest.fit(X_train, y_train)
rf_acc = randomforest.score(X_test, y_test)
print("Random forest test accuracy:", rf_acc)

# Neural network

# One-hot encode labels
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
    
num_features = features.shape[1]
num_classes = len(np.unique(labels))

print(y_train_onehot.shape, y_test_onehot.shape)

model = Sequential()
model.add(Dense(num_features, input_dim=num_features, init='normal', activation='relu'))
model.add(Dense(num_classes, init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), nb_epoch=100, batch_size=250, verbose=0)
loss, nn_acc = model.evaluate(X_test, y_test_onehot)

print("\nNeural network accuracy:", nn_acc)


print("Testing classifiers with 20 random train-test splits")
results = {}
results['knn'] = {}
results['svm'] = {}
results['dt'] = {}
results['rf'] = {}
results['nn'] = {}
results['gbt'] = {}

for result in results:
    results[result]['accuracy'] = []
    results[result]['precision'] = []
    results[result]['recall'] = []
    results[result]['f'] = []
    results[result]['train_time'] = []
    results[result]['test_time'] = []

classifiers = {
    'knn' : KNeighborsClassifier(n_neighbors=7),
    'svm' : SVC(kernel='rbf', C=2, gamma=0.02, probability=True),
    'dt'  : DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=1, max_leaf_nodes=(len(X_train) - 1)),
    'rf' : RandomForestClassifier(min_samples_split=10, min_samples_leaf=1, max_leaf_nodes=(len(X_train) - 1)),
    'gbt' : GradientBoostingClassifier()
}

for i in range(20):
    idx = np.random.permutation(len(features_scaled))
    features_scaled = features_scaled[idx]
    labels = labels[idx]

    X_train = features_scaled[:half]
    X_test = features_scaled[half:]
    y_train = labels[:half]
    y_test = labels[half:]    
    for clf_name in classifiers:
        clf = classifiers[clf_name]
        
        train_start = time.time()
        clf.fit(X_train, y_train)
        train_end = time.time()
        results[clf_name]['train_time'].append(train_end - train_start)
        
        test_start = time.time()
        clf.predict(X_test)
        test_end = time.time()
        results[clf_name]['test_time'].append(test_end - test_start)     
        
        _, accuracy = cross_val_score(clf, features_scaled, labels, cv=2, scoring='accuracy')
        _, precision = cross_val_score(clf, features_scaled, labels, cv=2, scoring='precision')
        _, recall = cross_val_score(clf, features_scaled, labels, cv=2, scoring='recall')
        _, f = cross_val_score(clf, features_scaled, labels, cv=2, scoring='f1')
        results[clf_name]['accuracy'].append(accuracy)
        results[clf_name]['precision'].append(precision)
        results[clf_name]['recall'].append(recall)
        results[clf_name]['f'].append(f)
        
    y_train_onehot = np_utils.to_categorical(y_train)
    y_test_onehot = np_utils.to_categorical(y_test)
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    
    model = Sequential()
    model.add(Dense(num_features, input_dim=num_features, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    train_start = time.time()
    model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), nb_epoch=100, batch_size=50, verbose=0)
    train_end = time.time()
    
    loss, nn_acc = model.evaluate(X_test, y_test_onehot)
    
    test_start = time.time()
    y_pred = model.predict(X_test)
    test_end = time.time()
    precision = average_precision_score(y_test_onehot, y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    recall = recall_score(y_test, y_pred)    
    f1 = f1_score(y_test, y_pred)    
    
    results['nn']['accuracy'].append(nn_acc)
    results['nn']['precision'].append(precision)
    results['nn']['recall'].append(recall)
    results['nn']['f'].append(f1)
    results['nn']['train_time'].append(train_end - train_start)
    results['nn']['test_time'].append(test_end - test_start)        

for clf_name in results:
    print(clf_name)
    print("Mean")
    print(pd.DataFrame.from_dict(results[clf_name]).mean())
    print("Standard deviation")
    print(pd.DataFrame.from_dict(results[clf_name]).std())
