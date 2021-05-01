#########################################################################
# breast_cancer
#########################################################################

# (1) 모듈 import
#####
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# (2) 데이터 준비
#####
breast_cancer = load_breast_cancer()
breast_cancer.keys()

# (3) 데이터 이해하기
#####
# Feature data 지정
breast_cancer_data = breast_cancer.data

# label data 지정
breast_cancer_label = breast_cancer.target

# target name 출력  
breast_cancer.target_names

# describe???
df = breast_cancer

# (4) train, test 데이터 분리
#####
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data,
                                                    breast_cancer_label,
                                                    test_size = 0.2,
                                                    random_state=10)

# (5) 다양한 모델로 학습시켜보기
#####

# 1. Decision Tree 사용
###
# 모델 불러오기
from sklearn.tree import DecisionTreeClassifier

# 인스턴스 생성
decision_tree = DecisionTreeClassifier(random_state=10)

# 학습시키기
decision_tree.fit(X_train, y_train)

# 모델 평가하기
y_pred_DT = decision_tree.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_DT = accuracy_score(y_test, y_pred_DT)
accuracy_DT     # 0.86

conf_mat_DT = confusion_matrix(y_test, y_pred_DT)
conf_mat_DT
#   [[36,  3],
#    [13, 62]]

print(classification_report(y_test, y_pred_DT))
#             precision    recall  f1-score   support

#            0       0.73      0.92      0.82        39
#            1       0.95      0.83      0.89        75

#     accuracy                           0.86       114
#    macro avg       0.84      0.87      0.85       114
# weighted avg       0.88      0.86      0.86       114


# 2. Random Forest 사용
from sklearn.ensemble import RandomForestClassifier

# 인스턴스 생성
random_forest = RandomForestClassifier(random_state=10)

# 학습시키기
random_forest.fit(X_train, y_train)

# 모델 평가하기
y_pred_RF = random_forest.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_RF = accuracy_score(y_test, y_pred_RF)
accuracy_RF     # 0.98

conf_mat_RF = confusion_matrix(y_test, y_pred_RF)
conf_mat_RF
# [[39,  0],
#  [ 2, 73]]

print(classification_report(y_test, y_pred_RF))
#              precision    recall  f1-score   support

#            0       0.95      1.00      0.97        39
#            1       1.00      0.97      0.99        75

#     accuracy                           0.98       114
#    macro avg       0.98      0.99      0.98       114
# weighted avg       0.98      0.98      0.98       114


# 3. SVM 사용
# 모델 불러오기
from sklearn import svm

# 인스턴스 생성
svm_model = svm.SVC()

# 모델 학습시키기
svm_model.fit(X_train, y_train)

# 모델 평가하기
y_pred_SVM = svm_model.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
accuracy_SVM    # 0.92

conf_mat_SVM = confusion_matrix(y_test, y_pred_SVM)
conf_mat_SVM
#   [[32,  7],
#    [ 2, 73]]

print(classification_report(y_test, y_pred_SVM))
#              precision    recall  f1-score   support

#            0       0.94      0.82      0.88        39
#            1       0.91      0.97      0.94        75

#     accuracy                           0.92       114
#    macro avg       0.93      0.90      0.91       114
# weighted avg       0.92      0.92      0.92       114


# 4. SGD 사용
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
accuracy_SGD   # 0.90

conf_mat_SGD = confusion_matrix(y_test, y_pred_SGD)
conf_mat_SGD
#    [[35,  4],
#     [ 4, 71]]

print(classification_report(y_test, y_pred_SGD))
#              precision    recall  f1-score   support

#            0       0.90      0.90      0.90        39
#            1       0.95      0.95      0.95        75

#     accuracy                           0.93       114
#    macro avg       0.92      0.92      0.92       114
# weighted avg       0.93      0.93      0.93       114


# Logistic Regression 사용
###
# 5. 모델 불러오기
from sklearn.linear_model import LogisticRegression

# 인스턴스 생성
logistic_model = LogisticRegression()

# 학습시키기
logistic_model.fit(X_train, y_train)

# 모델 평가하기
y_pred_LR = logistic_model.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_LR = accuracy_score(y_test, y_pred_LR)
accuracy_LR     # 0.93

conf_mat_LR = confusion_matrix(y_test, y_pred_LR)
conf_mat_LR
#   [[36,  3],
#    [ 5, 70]]

print(classification_report(y_test, y_pred_LR))
#              precision    recall  f1-score   support

#            0       0.88      0.92      0.90        39
#            1       0.96      0.93      0.95        75

#     accuracy                           0.93       114
#    macro avg       0.92      0.93      0.92       114
# weighted avg       0.93      0.93      0.93       114


# (6) 모델 평가해보기
#####

아래의 5종 모델 활용.
1. 의사결정나무 (DecisionTree)                       # 86
2. 랜덤 포레스트 (Random Forest)                     # 98
3. 서포트 벡터 머신 (Support Vector Machine)         # 92
4. 확률적 경사하강법 (Stochastic Gradient Descent)    # 92
5. 로지스틱 회귀 (Logistic Regression)               # 90
