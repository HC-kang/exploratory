#########################################################################
# wine
#########################################################################

# (1) 모듈 import 하기
#####
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# (2) 데이터 준비하기
#####
wine = load_wine()
wine.keys()

# (3) 데이터 이해하기
#####
wine.data.shape
wine.target
wine.target_names
wine.feature_names

# Feature Data 지정
wine_data = wine.data

# label data 지정
wine_label = wine.target

# target name 출력
wine.target_names

# 데이터 describe 해보기
wine_data

# (4) train, test 데이터 분리
#####
X_train, X_test, y_train, y_test = train_test_split(wine_data,
                                                    wine_label,
                                                    test_size=0.2,
                                                    random_state=10)

# (5) 다양한 모델로 학습시키기
#####

# 1. Decision Tree 사용해보기
###
# 모델 불러오기
from sklearn.tree import DecisionTreeClassifier

# 인스턴스 생성
decision_tree = DecisionTreeClassifier(random_state=10)

# 학습시키기
decision_tree.fit(X_train, y_train)

# 모델 평가하기
y_pred_DT = decision_tree.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인하기
accuracy_DT = accuracy_score(y_test, y_pred_DT)
accuracy_DT #0.94

conf_mat_DT = confusion_matrix(y_test, y_pred_DT)
conf_mat_DT
#       [[10,  0,  0],
#        [ 0, 16,  2],
#        [ 0,  0,  8]]

print(classification_report(y_test, y_pred_DT))

print(classification_report(y_test, y_pred_DT))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        10
#            1       1.00      0.89      0.94        18
#            2       0.80      1.00      0.89         8

#     accuracy                           0.94        36
#    macro avg       0.93      0.96      0.94        36
# weighted avg       0.96      0.94      0.95        36


# 2. Random Forest 사용
###
# 모델 불러오기
from sklearn.ensemble import RandomForestClassifier

# 인스턴스 생성
random_forest = RandomForestClassifier(random_state = 10)

# 학습시키기
random_forest.fit(X_train, y_train)

# 모델 평가하기
y_pred_RF = random_forest.predict(X_test)

# 정확도, 혼동행렬, 분류보고서 확인
accuracy_RF = accuracy_score(y_test, y_pred_RF)
accuracy_RF # 0.94

conf_mat_RF = confusion_matrix(y_test, y_pred_RF)
conf_mat_RF
#       [[10,  0,  0],
#        [ 0, 16,  2],
#        [ 0,  0,  8]]

print(classification_report(y_test, y_pred_RF))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        10
#            1       1.00      0.89      0.94        18
#            2       0.80      1.00      0.89         8

#     accuracy                           0.94        36
#    macro avg       0.93      0.96      0.94        36
# weighted avg       0.96      0.94      0.95        36


# 3. SVM 사용
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
accuracy_SVM    # 0.69

conf_mat_SVM = confusion_matrix(y_test, y_pred_SVM)
conf_mat_SVM
#       [[ 7,  0,  3],
#        [ 1, 13,  4],
#        [ 0,  3,  5]]

print(classification_report(y_test, y_pred_SVM))
#               precision    recall  f1-score   support

#            0       0.88      0.70      0.78        10
#            1       0.81      0.72      0.76        18
#            2       0.42      0.62      0.50         8

#     accuracy                           0.69        36
#    macro avg       0.70      0.68      0.68        36
# weighted avg       0.74      0.69      0.71        36


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
accuracy_SGD    # 0.67

conf_mat_SGD = confusion_matrix(y_test, y_pred_SGD)
conf_mat_SGD
#        [[ 8,  0,  2],
#        [ 1, 10,  7],
#        [ 1,  1,  6]]

print(classification_report(y_test, y_pred_SGD))
#               precision    recall  f1-score   support

#            0       0.80      0.80      0.80        10
#            1       0.91      0.56      0.69        18
#            2       0.40      0.75      0.52         8

#     accuracy                           0.67        36
#    macro avg       0.70      0.70      0.67        36
# weighted avg       0.77      0.67      0.68        36


# 5. Logistic Regression 사용
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
accuracy_LR     # 0.92

conf_mat_LR = confusion_matrix(y_test, y_pred_LR)
conf_mat_LR
#        [[10,  0,  0],
#        [ 1, 15,  2],
#        [ 0,  0,  8]]

print(classification_report(y_test, y_pred_LR))
#             precision    recall  f1-score   support

#            0       0.91      1.00      0.95        10
#            1       1.00      0.83      0.91        18
#            2       0.80      1.00      0.89         8

#     accuracy                           0.92        36
#    macro avg       0.90      0.94      0.92        36
# weighted avg       0.93      0.92      0.92        36


# (6) 모델 평가해보기
#####

아래의 5종 모델 활용.
1. 의사결정나무 (DecisionTree)                       # 94 [ㅇ]
2. 랜덤 포레스트 (Random Forest)                     # 94 [ㅇ]
3. 서포트 벡터 머신 (Support Vector Machine)         # 69 
4. 확률적 경사하강법 (Stochastic Gradient Descent)    # 67
5. 로지스틱 회귀 (Logistic Regression)               # 92
