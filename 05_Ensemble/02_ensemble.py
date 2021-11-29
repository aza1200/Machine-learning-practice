from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mnist = datasets.load_digits()
features, labels = mnist.data, mnist.target
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2)

dtree = tree.DecisionTreeClassifier(criterion="gini",max_depth=8,max_features=32,random_state=35)
dtree = dtree.fit(x_train,y_train)
dtree_predicted = dtree.predict(x_test)

knn = KNeighborsClassifier(n_neighbors=299).fit(x_train,y_train)
knn_predicted = knn.predict(x_test)


svm = SVC(kernel='rbf',C=0.1,gamma=0.003,probability=True,random_state=35).fit(x_train,y_train)
svm_predicted = svm.predict(x_test)

print("[accuracy]")
print("d-tree : ",accuracy_score(y_test,dtree_predicted))
print("knn : ",accuracy_score(y_test,knn_predicted))
print("svm : ",accuracy_score(y_test,svm_predicted))

svm_proba = svm.predict_proba(x_test)
print(svm_proba[0:2])


voting_clf = VotingClassifier(estimators= [
    ('deciison_tree',dtree),('knn',knn),('svm',svm)],
    weights= [1,1,1],voting='hard').fit(x_train,y_train)

hard_voting_predicted = voting_clf.predict(x_test)
print(accuracy_score(y_test,hard_voting_predicted))

voting_clf = VotingClassifier(estimators = [
    ('decision_tree',dtree),('knn',knn),('svm',svm)],
    weights = [1,1,1],voting = 'soft').fit(x_train,y_train)

soft_voting_predicted = voting_clf.predict(x_test)
accuracy_score(y_test,soft_voting_predicted)


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
plt.bar(x,height=[
    accuracy_score(y_test, dtree_predicted ),
    accuracy_score(y_test, knn_predicted ),
    accuracy_score(y_test, svm_predicted ),
    accuracy_score(y_test, hard_voting_predicted ),
    accuracy_score(y_test, soft_voting_predicted )
])

plt.xticks(x,['decision tree','knn','svm','hard voting','soft voting'])
plt.show()
