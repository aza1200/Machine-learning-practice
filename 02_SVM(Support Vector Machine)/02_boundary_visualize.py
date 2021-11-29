from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np



def svc_param_selection(X,y,nfolds):
    svm_parameters = [
        {'kernel' : ['rbf'],
          'gamma' : [0.00001,0.0001,0.001,0.01,0.1,1],
              'C' : [0.01,0.1,1,10,100,1000]
        }
    ]


    # 사이킷런에서 제공하는 GridSearchCV 를 사용해 최적의 파라미터를 구함
    clf = GridSearchCV(SVC(),svm_parameters,cv=10)
    clf.fit(X,y)

    print(clf.best_params_)
    return clf

def main():
    df = pd.read_csv('data.csv')
    df.drop(['2P', 'AST', 'STL'], axis=1, inplace=True)
    train, test = train_test_split(df, test_size=0.2)

    X_train = train[['3P', 'BLK']]
    y_train = train[['Pos']]

    clf = svc_param_selection(X_train,y_train.values.ravel(),10)


    #시각화할 비용 후보들을 저장
    C_candidates = []
    C_candidates.append(clf.best_params_['C']* 0.01)
    C_candidates.append(clf.best_params_['C'])
    C_candidates.append(clf.best_params_['C']* 100)

    #시각화할 감마 후보들을 저장
    gamma_candidates = []
    gamma_candidates.append(clf.best_params_['gamma']* 0.01)
    gamma_candidates.append(clf.best_params_['gamma'])
    gamma_candidates.append(clf.best_params_['gamma']* 100)

    # 3점 슛과 블로킹 횟수로 학습
    X = train[['3P', 'BLK']]
    Y = train['Pos'].tolist()

    #시각화를 위해 센터(C)와 슈팅가드(5G) 를 숫자로 표현
    position = []
    for gt in Y:
        if gt == 'C':
            position.append(0)
        else:
            position.append(1)

    classifiers = []

    # 파라미터 후보들을 조합해서 학습된 모델들을 저장
    for C in C_candidates:
        for gamma in gamma_candidates:
            clf = SVC(C=C,gamma=gamma)
            clf.fit(X,Y)

            classifiers.append((C,gamma,clf))


    # 각 모델을 시각화
    plt.figure(figsize=(18,18))
    xx, yy = np.meshgrid(np.linspace(0,4,100),np.linspace(0,4,100))

    for (k,(C,gamma,clf)) in enumerate(classifiers):
        Z = clf.decision_function( np.c_[xx.ravel(),yy.ravel()] )
        Z = Z.reshape(xx.shape)

        plt.subplot(len(C_candidates),len(gamma_candidates),k+1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma),np.log10(C)), size='medium')

        plt.pcolormesh(xx,yy,-Z,cmap=plt.cm.RdBu)
        plt.scatter(X['3P'],X['BLK'],c=position,cmap=plt.cm.RdBu_r,edgecolors='k')

    plt.show()

if __name__ == '__main__':
    main()