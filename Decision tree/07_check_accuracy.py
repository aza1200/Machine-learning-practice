
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pandas as pd
import data
import graphviz
from sklearn.metrics import accuracy_score

train_df = pd.DataFrame(data.district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]

x_train = train_df[['longitude','latitude']]
y_train = train_df[['label']]

test_df = pd.DataFrame(data.dong_dict_list)
test_df = test_df[['dong','longitude','latitude','label']]

x_test = test_df[['longitude','latitude']]
y_test = test_df[['label']]

# Encode target labels with values between 0 and n_classes-1
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y_train)

clf = tree.DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=70).fit(x_train,y_encoded.ravel())

pred = clf.predict(x_test)
print('accuracy : '+str( accuracy_score(y_test.values.ravel(),le.classes_[pred])))

comparision = pd.DataFrame({'prediction':le.classes_[pred],
                            'gournd_truth':y_test.values.ravel()
                            })

