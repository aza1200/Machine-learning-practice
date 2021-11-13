from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pandas as pd
import data

train_df = pd.DataFrame(data.district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]

x_train = train_df[['longitude','latitude']]
y_train = train_df[['label']]

# Encode target labels with values between 0 and n_classes-1
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y_train)

clf = tree.DecisionTreeClassifier(random_state=35).fit(x_train,y_encoded)


def display_decision_surface(clf,X,y):
    x_min = X.longitude.min() - 0.01
    x_max = X.longitude.max() + 0.01
    y_min = X.latitude.min() - 0.01
    y_max = X.latitude.max() + 0.01

    n_classes = len(le.classes_)
    plot_colors = 'rywb'
    plot_step = 0.001

    # 좌표 설계
    xx, yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
                         np.arange(y_min,y_max,plot_step))

    # 각각의 이차원 좌표에 해당하는거 예측
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    # 1차원 배열 2차원 배열로 다시 돌아트림
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu)

    for i, color in zip(range(n_classes),plot_colors):
        idx = np.where( y == i )
        plt.scatter(X.loc[idx].longitude,X.loc[idx].latitude,
                    c=color,label=le.classes_[i],cmap=plt.cm.RdYlBu,edgecolors='black',
                    s=200)

    plt.title("Decision surface of a decision tree",fontsize=16)
    plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0,fontsize=14)
    plt.xlabel('longitude',fontsize= 16)
    plt.ylabel('latitude',fontsize = 16)
    plt.rcParams["figure.figsize"] = [7,5]
    plt.rcParams["font.size"] = 14
    plt.rcParams['xtick.labelsize'] =14
    plt.rcParams['ytick.labelsize'] = 14
    plt.show()


display_decision_surface(clf,x_train,y_encoded)