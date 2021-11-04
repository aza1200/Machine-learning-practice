import pandas as pd

# Player : 선수이름
# Pos : 포지션
# 3P : 한경기 평균 3점슛 성공 횟수
# 2P : 한경기 평균 2점슛 성공 횟수
# TRB : 한 경기 평균 리바운드 성공 횟수
# AST : 한경기 평균 어시스트 성공 횟수
# STL : 한 경기 평균 스틸 성공 횟수
# BLK : 한 경기 평균 블로킹 성공 횟수


def main():
    df = pd.read_csv('data.csv')
    # 샘플 확인print(df.head(n=5))

    print(df.Pos.value_counts())


if __name__ == '__main__':
    main()