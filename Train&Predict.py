#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from numpy.core.multiarray import ndarray
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class InternalRandomForest(object):
    """
    Random Forest Training Class using previously preparated dataset.
    """

    @staticmethod
    def modeltrain(features: ndarray, target: ndarray, number_trees: int) -> RandomForestClassifier:
        """
        Method to create and fit a new trained RF model by a known number of trees
        Args:
            features                         (ndarray): Array of dataset features (samples).
            target                           (ndarray): Array of dataset targets (species).
            number_trees                     (int): Number of trees to train classifier.
        Returns:
            trained_model                    (object): Trained RandomForestClassifier object.
        """

        trained_model = RandomForestClassifier(number_trees)
        trained_model.fit(features, target)

        return trained_model

    @staticmethod
    def getbestnumberoftrees(features: ndarray, target: ndarray, limit:int) -> tuple:
        """
        Method to know the number of trees that generate the best training
        using the ACCURACY method of Random Forest
        Args:
            features                         (ndarray): Array of dataset features (samples).
            target                           (ndarray): Array of dataset targets (species).
            limit                            (int): Limit of number of trees to train and obtain best number trees.
        Returns:
            bestNumberTrees                  (int): Best number of trees in training.
            accuracyList                     (list): List of accuracies obtained during training sets.
            best_model                       (object): Model trained with best number of trees.
        """

        # Defining the initial accuracy value to compare with different number of trees in training
        accuracy = 0
        accuracyList = []

        for n in range(1, limit+1, 1):
            # Training
            trained_model = InternalRandomForest.modeltrain(features, target, n)

            # Calculating the percentual accuracy of the training
            accuracy_t = accuracy_score(target, trained_model.predict(features), normalize=True)

            # Build accuracy array for this set of number of trees
            accuracyList.append(accuracy_t)

            # Verifying if the current training is better than the last one
            if accuracy_t > accuracy:
                bestNumberTrees = n
                accuracy = accuracy_t

        # Obtain best trained model
        best_model = InternalRandomForest.modeltrain(features, target, bestNumberTrees)

        return bestNumberTrees, accuracyList, best_model

    @staticmethod
    def gethitscm(features: ndarray, target: ndarray, trained_model) -> tuple:
        """
        Method to know the global hits and the hits of the true objects using
        the Confusion Matrix
        Args:
            features                         (ndarray): Array of dataset features (samples).
            target                           (ndarray): Array of dataset targets (species).
            trained_model                    (object): Trained RandomForestClassifier object.
        Returns:
            global_hits                      (list): Global hits in CM.
            true_hits                        (list): True hits in CM.
        """
        # Generating the Confusion Matrix
        predictions = trained_model.predict(features)

        cm = confusion_matrix(target, predictions)

        # Calculating the global hits
        global_hits = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
        true_hits = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        return (cm, global_hits, true_hits)

    @staticmethod
    def getaccuracy(features: ndarray, target: ndarray, trained_model) -> float:
        """
        Method to get accuracy from a previously trained model, applying features and samples from test dataset.
        Args:
            features                         (ndarray): Array of test dataset features (samples).
            target                           (ndarray): Array of test dataset targets (species).
            trained_model                    (object): Trained RandomForestClassifier object.
        Returns:
            accuracy                         (float): Accuracy of Test
        """
        predictions = trained_model.predict(features)

        accuracy = accuracy_score(target, predictions, normalize=True)

        return accuracy


# In[82]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

teste = lambda x: x.strip("[]").replace("'", "").split(", ")
datasoil = pd.read_csv('TCORTE1.csv',index_col=False, sep=";", converters={'Soil': teste})
Targets = [list(map(float, data)) for data in datasoil['Soil']]
Targets = np.array(Targets)

q = pd.DataFrame(Targets)
Target = q.iloc[:,0]#escolher dado do solo aqui*******************

le = LabelEncoder()
le.fit(Target)
Target = le.transform(Target)


# In[113]:


data = pd.read_csv('TCORTE1.csv', index_col=False, sep=";")

k=np.array(data['Kurtosis']).reshape(len(data['X']),1)
s=np.array(data['Skewness']).reshape(len(data['X']),1)
datacolor = pd.read_csv('TCORTE1.csv',index_col=False, sep=";", converters={'Color': teste})
color = np.array([list(map(float, data)) for data in datacolor['Color']])
Features = np.concatenate((k,s,color),axis=1)


# In[72]:


import time
from sys import path, platform
from os import getcwd
from numpy import load
from numpy import arange
from numpy import linspace
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


# In[73]:


rf = InternalRandomForest()


# In[109]:


treeLimit = 50
bestNumberTrees, accuracy, bestModel = rf.getbestnumberoftrees(Features, Target, treeLimit)


# In[110]:


bestNumberTrees


# In[111]:


accuracy


# In[112]:


bestModel


# In[ ]:


predictions = rf.predict(data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[115]:


train = pd.read_csv('Train.csv', index_col=False, sep=";")
train.head()


# In[116]:


len(train['X'])


# In[ ]:




