from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pandas as pd
import data

train_df = pd.DataFrame(data.district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]

test_df = pd.DataFrame(data.dong_dict_list)
test_df = test_df[['dong','longitude','latitude','label']]
x_train = train_df[['longitude','latitude']]
y_train = train_df[['label']]

x_test = test_df[['longitude','latitude']]
y_test = test_df[['label']]

# Encode target labels with values between 0 and n_classes-1
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y_train)
clf = tree.DecisionTreeClassifier(random_state=35).fit(x_train,y_encoded)