from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd



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


if __name__ == '__main__':
    main()