# 시각화를 위해 pandas 임포트
import pandas as pd
# iris 데이터는 sklearn 에서 직접 로드 가능
from sklearn.datasets import load_iris
# train_test_split 사용시 손쉽게 데이터 나눌수 있음
from sklearn.model_selection import train_test_split
# 분류 성능을 측정하기 위해 metrics 의 accuracy_score 를 임포트한다.
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

# iris 데이터 불러옴
dataset = load_iris()
# pandas의 데이터프레임으로 데이터를 저장
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# 레이블(타깃) 을 데이터프레임에 저장
df['target'] = dataset.target
# 숫자 형태의 레이블을 이해를 돕기위해 문자로 변경
df.target = df.target.map({0:"setosa",1:"versicolor",2:"virginica"})

setosa_df = df[df.target == "setosa"]
versicolor_df = df[df.target == "versicolor"]
virginica_df = df[df.target == "virginica"]


X_train,X_test,y_train,y_test =\
    train_test_split(dataset.data,dataset.target,test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(y_test,predicted))

print(accuracy_score(y_test,predicted))

