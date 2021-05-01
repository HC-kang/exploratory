
#########################################################################
# digits 부터 해보기
#########################################################################

# (1) 모듈 import 하기
#####
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# (2) 데이터 준비
#####
digits = load_digits()
digits.keys()

# (3) 데이터 이해하기
#####
# Feature data 지정
digits_data = digits.data

# label data 지정
digits_label = digits.target

# target name 출력
digits.target_names

# ???digit을 describe 하라고???
digits_data

# (4) train, test 데이터 분리
#####
X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    digits_label,
                                                    test_size=0.2,
                                                    random_state=10)

# (5) 다양한 모델로 학습시켜보기
#####

# Decision Tree 사용
###
# 모델 불러오기
from sklearn.tree import DecisionTreeClassifier

# 인스턴스 생성
decision_tree = DecisionTreeClassifier(random_state = 10)

# 학습시키기
decision_tree.fit(X_train, y_train)

# 모델 평가하기
y_pred_DT = decision_tree.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_DT = accuracy_score(y_test, y_pred_DT)
accuracy_DT # 0.21

conf_mat_DT = confusion_matrix(y_test, y_pred_DT)
conf_mat_DT

print(classification_report(y_test, y_pred_DT))


# Random Forest 사용
###
# 모델 불러오기
from sklearn.ensemble import RandomForestClassifier

# 인스턴스 생성
random_forest = RandomForestClassifier(random_state=10)

# 학습시키기
random_forest.fit(X_train, y_train)

# 모델 평가하기
y_pred_RF = random_forest.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_RF = accuracy_score(y_test, y_pred_RF)
accuracy_RF # 0.97

conf_mat_RF = confusion_matrix(y_test, y_pred_RF)
conf_mat_RF

print(classification_report(y_test, y_pred_RF))


# SVM 사용
###
# 모델 불러오기
from sklearn import svm

# 인스턴스 생성
svm_model = svm.SVC()

# 학습시키기
svm_model.fit(X_train, y_train)

# 모델 평가하기
y_pred_SVM = svm_model.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
accuracy_SVM # 0.98

conf_mat_SVM = confusion_matrix(y_test, y_pred_SVM)
conf_mat_SVM

print(classification_report(y_test, y_pred_SVM))


# SGD 사용
###
# 모델 불러오기
from sklearn.linear_model import SGDClassifier

# 인스턴스 생성
sgd_model = SGDClassifier()

# 학습시키기
sgd_model.fit(X_train, y_train)

# 모델 평가하기
y_pred_SGD = sgd_model.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_SGD = accuracy_score(y_test, y_pred_SGD)
accuracy_SGD    # 0.938

conf_mat_SGD = confusion_matrix(y_test, y_pred_SGD)
conf_mat_SGD

print(classification_report(y_test, y_pred_SGD))


# Logistic Regresssion 사용
###
# 모델 불러오기
from sklearn.linear_model import LogisticRegression

# 인스턴스 생성
logistic_model = LogisticRegression()

# 학습시키기
logistic_model.fit(X_train, y_train)

# 모델 평가하기
y_pred_LR = logistic_model.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_LR = accuracy_score(y_test, y_pred_LR)
accuracy_LR     # 0.95

conf_mat_LR = confusion_matrix(y_test, y_pred_LR)
conf_mat_LR

print(classification_report(y_test, y_pred_LR))


# (6) 모델 평가해보기
#####
