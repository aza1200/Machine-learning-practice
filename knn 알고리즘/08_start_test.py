from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# Player : 선수이름
# Pos : 포지션
# 3P : 한경기 평균 3점슛 성공 횟수
# 2P : 한경기 평균 2점슛 성공 횟수
# TRB : 한 경기 평균 리바운드 성공 횟수
# AST : 한경기 평균 어시스트 성공 횟수
# STL : 한 경기 평균 스틸 성공 횟수
# BLK : 한 경기 평균 블로킹 성공 횟수

# 한플레이어의 Pos 예측이 목적임
def main():
    df = pd.read_csv('data.csv')

    #Pos,3P,TRB,BLK
    df.drop(['2P', 'AST', 'STL'], axis=1, inplace=True)
    # 열: axis = 1
    # 행: axis = 0

    #다듬어진 데이터 20% 테스트 데이터로 분류
    train,test = train_test_split(df,test_size=0.2)

    max_k_range = train.shape[0]//2
    k_list = []
    for i in range(3,max_k_range,2):
        k_list.append(i)

    cross_validation_scores = []
    x_train = train[['3P','BLK','TRB']]
    y_train = train[['Pos']]

    knn = KNeighborsClassifier(n_neighbors=3)
    # kNN 모델 학습
    knn.fit(x_train,y_train.values.ravel())

    x_test  = test[['3P','BLK','TRB']]
    y_test = test[['Pos']]

    #테스트 시작
    pred = knn.predict(x_test)

    #모델 예측 정확도(accuracy) 출력
    print("accuracy : "+ str( accuracy_score(y_test.values.ravel(),pred) ))

if __name__ == '__main__':
    main()