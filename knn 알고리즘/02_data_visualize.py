import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

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

    # sns.lmplot('STL','2P',data=df,fit_reg=False,
    #            scatter_kws={'s':150}, #좌표상의 점의 크기
    #            markers=['o','x'],
    #            hue="Pos") #예측값
    #
    # plt.title('STL and 2P in 2d plane')


    # sns.lmplot('BLK','3P',data=df,fit_reg=False,
    #            scatter_kws={'s':150}, #좌표상의 점의 크기
    #            markers=['o','x'],
    #            hue="Pos") #예측값
    #
    # plt.title('BLK and 3P in 2d plane')



    plt.show()
if __name__ == '__main__':
    main()