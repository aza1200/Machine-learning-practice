from sklearn.model_selection import train_test_split
import pandas as pd

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

    print(train.shape[0])
    print(test.shape[0])

if __name__ == '__main__':
    main()