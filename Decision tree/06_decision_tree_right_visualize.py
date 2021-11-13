from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pandas as pd
import data
import graphviz

train_df = pd.DataFrame(data.district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]

x_train = train_df[['longitude','latitude']]
y_train = train_df[['label']]

# Encode target labels with values between 0 and n_classes-1
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y_train)

clf = tree.DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=70).fit(x_train,y_encoded.ravel())




don_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(don_data)
graph.render("seoul")
don_data = tree.export_graphviz(clf,out_file=None,
                                feature_names=['longitude','latitude'],
                                class_names=['Gangbuk','Gangdong','Gangnam','Gangseo'],
                                filled=True,rounded=True,
                                special_characters=True)

graph = graphviz.Source(don_data)
graph

