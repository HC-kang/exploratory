from sklearn. datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

print(type(dir(iris)))
# dir()는 객체가 어떤 변수와 메서드를 가지고 있는지 나열함.

iris.keys()
# data, target, frame, target_names, DESCR, feature_names, filename 6가지 정보

iris_data = iris.data

print(iris_data.shape) # (150, 4)

iris.data[0]    # array([5.1, 3.5, 1.4, 0.2])

iris_label = iris.target
print(iris_label.shape) #(150,)
print(iris_label)

iris.target_names # array(['setosa', 'versicolor', 'virginica'])

print(iris.DESCR)

print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris.filename

import pandas as pd

print(pd.__version__)

iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)
iris_df

# adding columns 'label'
iris_df['label'] = iris.target
iris_df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_label,
                                                    test_size=0.2,
                                                    random_state=7)
# random_state 는 train data 와 test data의 분리 시 적용되는 랜덤성 결정?
# random_state = random_seed인듯.


print('X_train 개수 : ', len(X_train), ', X_test 개수 : ', len(X_test))

print('X_train data', X_train.shape)
print('X_test data', X_test.shape)
print('y_train data', y_train.shape)
print('y_test data', y_test.shape)

y_train
y_test
# 랜덤성 확인

#########
# 2-5
# 정답이 있으므로 이번 실습은 지도학습
# 지도학습은 두 종류로 나뉨, 분류와 회귀.
# 그중에서도 Iris 는 분류 문제.

# 그리고 그중에서 Decision Tree 모델 활용 예정

# 사이킷런에서 의사결정나무 모델 불러오기
from sklearn.tree import DecisionTreeClassifier

# 인스턴스 생성
decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)

# 학습시키기
decision_tree.fit(X_train, y_train)

# 2-6 모델 평가하기
y_pred = decision_tree.predict(X_test)

y_pred  # 모델의 예측
y_test  # 실제 답안

# 성능 평가를 위한 패키지 불러오기
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy    # 0.9
# 정확도 = 모델이 맞다고 한 것 중 진짜 정답인 확률

##################
# 2-7 다른 모델로 실습해보기


#################
# 의사결정나무
#################

# 필요한 모듈들 import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 데이터 준비
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

# train, test 데이터 분리
X_train, X_test, y_train,y_test = train_test_split(iris_data,
                                                   iris_label,
                                                   test_size=0.2,
                                                   random_state=25)

# 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))

# 랜덤스테이트 = 7
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         7
#            1       0.91      0.83      0.87        12
#            2       0.83      0.91      0.87        11

#     accuracy                           0.90        30
#    macro avg       0.91      0.91      0.91        30
# weighted avg       0.90      0.90      0.90        30

##### 뭐야 랜덤스테이트 바꾸니까 랜덤포레스트랑 똑같이 나옴??
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       0.92      0.92      0.92        13
#            2       0.88      0.88      0.88         8

#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30


# 그냥 반복,,


################
# 랜덤 포레스트
################

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_label,
                                                    test_size=0.2,
                                                    random_state=25)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
print(classification_report(y_test, y_pred)) # 정확도 0.93으로 향상

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       0.92      0.92      0.92        13
#            2       0.88      0.88      0.88         8

#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30


##################
# SVM(support vector machine)
##################

from sklearn import svm

svm_model = svm.SVC()
print(svm_model._estimator_type)

X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_label,
                                                    test_size=0.2,
                                                    random_state=25)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))

# svm, random_state = 25, 정확도 0.97. 제일 높네
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       1.00      0.92      0.96        13
#            2       0.89      1.00      0.94         8

#     accuracy                           0.97        30
#    macro avg       0.96      0.97      0.97        30
# weighted avg       0.97      0.97      0.97        30


####################
# SGDClassifier(stochastic gradient descent classifier) : 경사하강법
####################

from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
print(sgd_model._estimator_type)

X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_label,
                                                    test_size=0.2,
                                                    random_state=25)

sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))

# 경사하강법, random_state : 25 정확도 0.9
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       1.00      0.77      0.87        13
#            2       0.73      1.00      0.84         8

#     accuracy                           0.90        30
#    macro avg       0.91      0.92      0.90        30
# weighted avg       0.93      0.90      0.90        30


################
# Logistic Regression : 로지스틱 회귀 - 이건 근데 2개 분류 아닌가?
################

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
print(logistic_model._estimator_type)

X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_label,
                                                    test_size=0.2,
                                                    random_state=25)

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 로지스틱 회귀, 정확도 0.97
#              precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       1.00      0.92      0.96        13
#            2       0.89      1.00      0.94         8

#     accuracy                           0.97        30
#    macro avg       0.96      0.97      0.97        30
# weighted avg       0.97      0.97      0.97        30


##############
# 2-4 오차행렬_iris
# 2-8 내 모델은 얼마나 똑똑한가? 정확도에는 함정이 있다.

# 손글씨 자료 가져오기
from sklearn.datasets import load_digits

digits = load_digits()
digits.keys()

digits_data = digits.data
digits_data.shape   # (1797, 64)

digits_data[0]

import matplotlib.pyplot as plt
%matplotlib inline

# 이미지 불러보기
plt.imshow(digits.data[0].reshape(8,8), cmap = 'gray')
plt.axis('off')
plt.show()

# 여러 개 이미지 불러오기
for i in range(10):
    plt.subplot(2,5,1+i)
    plt.imshow(digits.data[i].reshape(8,8), cmap='gray')
    plt.axis('off')
plt.show()

digits_label = digits.target
print(digits_label.shape) #(1797,)
digits_label[:20]

# 정확도의 함정을 알아보기 위해, 3인지 여부만 알아보는 bool형태로.

new_label = [3 if i==3 else 0 for i in digits_label]
new_label[:20]


##################
# 의사결정나무 코드 3 판별기
##################

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 데이터 준비
digits = load_digits()
digits_data = digits.data
digits_label = new_label

# train, test 데이터 분리
X_train, X_test, y_train,y_test = train_test_split(digits_data,
                                                   digits_label,
                                                   test_size=0.2,
                                                   random_state=15)
# 아니 근데 모델이랑 데이터셋이 바뀌는데 왜자꾸 random_state를 바꾸는거야?

# 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=15)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy # 0.9388

# fake_pred 확인점검

fake_pred = [0] * len(y_pred)

accuracy_fake = accuracy_score(y_test, fake_pred)
accuracy_fake  #0.925
# 모델 성능을 평가할 때, 특히 분포가 고르지 못한 표본을 다룰 때에 유의해야 함.


####################
# 2-9 정답과 오답에도 종류가 있다

from sklearn.metrics import confusion_matrix

# 모델이 3으로 예측한 결과
confusion_matrix(y_test, y_pred)
# ([[320,  13],
#  [  9,  18]])

# 가짜 결과
confusion_matrix(y_test, fake_pred)
# ([[333,   0],
#   [ 27,   0]])

# Precision, Recall, f1 score 확인하기
from sklearn.metrics import classification_report

# 모델이 3으로 예측한 결과
print(classification_report(y_test, y_pred))
#              precision    recall  f1-score   support

#            0       0.97      0.96      0.97       333
#            3       0.58      0.67      0.62        27

#     accuracy                           0.94       360
#    macro avg       0.78      0.81      0.79       360
# weighted avg       0.94      0.94      0.94       360

# 가짜 결과
print(classification_report(y_test, fake_pred))
#              precision    recall  f1-score   support

#            0       0.93      1.00      0.96       333
#            3       0.00      0.00      0.00        27

#     accuracy                           0.93       360
#    macro avg       0.46      0.50      0.48       360
# weighted avg       0.86      0.93      0.89       360

# 정확도 확인하기
accuracy_score(y_test, y_pred)      # 0.93888
accuracy_score(y_test, fake_pred)   # 0.925
