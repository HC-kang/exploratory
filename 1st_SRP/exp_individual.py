# 1-1 인공지능과 가위바위보 하기
# 1-2 데이터를 준비하자

# 텐서플로우, 케라스 소환
import tensorflow as tf
from tensorflow import keras

# 넘파이, 맷플롯립 소환
import numpy as np
import matplotlib.pyplot as plt

# 걍 한번 버전 확인,,
print(tf.__version__)

# 케라스 데이터셋 변수 mnist로 저장
mnist = keras.datasets.mnist

# MNIST 데이터를 로드. 다운로드 하지 않았다면 자동으로 다운로드까지 진행
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print('x_train 의 크기 :',len(x_train))     # x_train 배열의 크기를 출력
print('y_train 의 크기 :',len(y_train))     # y_train 배열의 크기를 출력
print('x_test 의 크기 :',len(x_test))       # x_test 배열의 크기를 출력
print('x_test 의 크기 :',len(y_test))       # y_test 배열의 크기를 출력


plt.imshow(x_train[1], cmap=plt.cm.binary) # 이걸 왜 굳이 바이너리로 뽑은건지??
plt.show()

print(y_train[1]) # x_train[1]에 해당하는 라벨값 보여주기.

# index 에 0에서 59999 사이 숫자를 지정해 보세요.
index = 10000 # 임의지정 숫자
plt.imshow(x_train[index], cmap=plt.cm.binary) # 여기도 또 바이너리,, 왜?
plt.show()
print((index+1), '번째 이미지의 숫자는 바로 ', y_train[index],'입니다.')
# 인덱스는 시작이 0이니 당연히 +1, 이걸 은근히 계속 실수한다,,


print(x_train.shape) # 이미지 자료이니 사이즈는 (60000, 28, 28)
print(y_train.shape) # 라벨이다보니 그냥 60000,

print(x_test.shape) # 이미지 자료이니 사이즈는 (10000, 28, 28)
print(y_test.shape) # 역시 그냥 10000,

# x_train 배열 내부의 최소, 최대값. 픽셀 당 1바이트로 0~255까지 256개 표현가능
print('최소값 : ',np.min(x_train), '최대값 : ',np.max(x_train))

# 최대값으로 나눠주어 정규화. 아니 근데 왜 .0이야
x_train_norm, x_test_norm = x_train /255.0, x_test / 255.0
print('최소값 : ',np.min(x_train_norm), '최대값 : ', np.max(x_train_norm))


#####################################
# 1-3 딥러닝 네트워크 설계

model = keras.models.Sequential()
# Conv2D, filter는 커널의 개수,,
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape = (28,28,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32,(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

print('model에 추가된 Layer 개수 : ', len(model.layers))

model.summary()

############################
# 1-4 딥러닝 네트워크 학습시키기

print("Before reshape - x_train_norm shape : {}".format(x_train_norm.shape))
print('Before reshape - x_test_norm shape : {}'.format(x_test_norm.shape))
# 3차원.


x_train_reshaped = x_train_norm.reshape(-1, 28, 28, 1)
x_test_reshaped = x_test_norm.reshape(-1, 28, 28, 1)
# 데이터 개수에 -1을 쓰면 자동 계산
# 3차원에서 4차원으로 바꿔줌. 왜??

print("After Reshape - x_train_reshaped shape : {}".format(x_train_reshaped.shape))
print("After Reshape - x_test_reshaped shape : {}".format(x_test_reshaped.shape))
# 4차원

# 컴파일
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# 학습 시키기.
model.fit(x_train_reshaped, y_train, epochs=10)

###############################
# 1-5 얼마나 잘 만들었는지 확인하기 - 10회차 정확도 99.64%

# 실험용 데이터로 점수 확인 - 정확도 98.55%
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test, verbose = 2)

print('test_loss : {}'.format(test_loss))               # 0.0476800762116909
print('test_accuracy : {}'.format(test_accuracy))       # 0.9854999780654907


# 눈으로 확인해보기
predicted_result = model.predict(x_test_reshaped) # model이 추론한 확률값.
predicted_labels = np.argmax(predicted_result, axis = 1)

idx = 0 # 1번째 x_test 를 살펴보자.
print('model.predict()결과 : ', predicted_result[idx])
print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
print('실제 데이터의 라벨 : ', y_test[idx])

plt.imshow(x_test[idx], cmap=plt.cm.binary)
plt.show()

# model이 추론해 낸 숫자와 실제 라벨의 값이 다른 경우는?
# 눈으로 확인하기
import random
wrong_predict_list = []

for i, _ in enumerate(predicted_labels):
    # i번째 test_labels과 y_test가 다른 경우만 모으기.
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

# wrong_predict_list 에서 랜덤하게 5개만 뽑아봅시다.
samples = random.choices(population=wrong_predict_list, k=5)

for n in samples:
    print("예측확률분포 : " + str(predicted_result[n]))
    print("라벨 : " + str(y_test[n]) + ", 예측결과 : " + str(predicted_labels[n]))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()

########################################
# 1-6 더 좋은 네트워크 만들기

# 바꿔 볼 수 있는 하이퍼파라미터들

n_channel_1 = 16
n_channel_2 = 32
n_dense = 32
n_train_epoch=10

model=keras.models.Sequential()

model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation= 'relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# 모델 훈련 - 정확도 99.66%
model.fit(x_train_reshaped, y_train, epochs=n_train_epoch)


# 모델 시험 - 정확도 98.9%

test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test, verbose=2)
print('test_loss : {}'.format(test_loss))
print('test_accuracy : {}'.format(test_accuracy))



###################################################
# 본격 가위바위보
from PIL import Image
import os, glob

# 가위 
images = glob.glob('/Users/heechankang/projects/pythonworkspace/exploratory/test_data/scissors' + '/*.jpg')
# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size = (28, 28)
for img in images:
    old_img = Image.open(img)
    new_img = old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,'JPEG')
print('가위 이미지 resize 완료!')

# 바위
images = glob.glob('/Users/heechankang/projects/pythonworkspace/exploratory/test_data/rocks' + '/*.jpg')
# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size = (28, 28)
for img in images:
    old_img = Image.open(img)
    new_img = old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,'JPEG')
print('바위 이미지 resize 완료!')

# 보
images = glob.glob('/Users/heechankang/projects/pythonworkspace/exploratory/test_data/papers' + '/*.jpg')
# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size = (28, 28)
for img in images:
    old_img = Image.open(img)
    new_img = old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,'JPEG')
print('보 이미지 resize 완료!')

###########################

def load_data(img_path):                        # 데이터 불러오는 함수
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data = 300 # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size = 28
    color = 3

    # 이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color, dtype=np.int32).reshape(number_of_data, img_size, img_size, color)
    labels = np.zeros(number_of_data,dtype=np.int32)

    idx = 0
    for file in glob.iglob(img_path+'/scissors/*.jpg'):
        img=np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0 # 가위 : 0
        idx = idx+1

    for file in glob.iglob(img_path+'/rocks/*.jpg'):
        img=np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1 # 바위 : 1
        idx = idx+1

    for file in glob.iglob(img_path+'/papers/*.jpg'):
        img=np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2 # 보 : 2
        idx = idx+1

    print('학습 데이터(x_train)의 이미지 개수는',idx,'입니다.')
    return imgs, labels



def load_test_data(img_path):                        # 데이터 불러오는 함수
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data = 70 # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size = 28
    color = 3

    # 이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color, dtype=np.int32).reshape(number_of_data, img_size, img_size, color)
    labels = np.zeros(number_of_data,dtype=np.int32)

    idx = 0
    for file in glob.iglob(img_path+'/scissors/*.jpg'):
        img=np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0 # 가위 : 0
        idx = idx+1

    for file in glob.iglob(img_path+'/rocks/*.jpg'):
        img=np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1 # 바위 : 1
        idx = idx+1

    for file in glob.iglob(img_path+'/papers/*.jpg'):
        img=np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2 # 보 : 2
        idx = idx+1

    print('학습 데이터(x_train)의 이미지 개수는',idx,'입니다.')
    return imgs, labels

# 데이터 불러올 주소 설정
image_dir_path = '/Users/heechankang/projects/pythonworkspace/exploratory/samples'
train_image_dir_path = '/Users/heechankang/projects/pythonworkspace/exploratory/test_data'

# 불러온 주소를 가지고 load_data함수 실행. 데이터 가져오기.
(x_train, y_train) = load_data(image_dir_path)

(x_test, y_test) = load_test_data(train_image_dir_path)

# data 크기 확인
print('x_train 의 크기 :',len(x_train))     # x_train 배열의 크기를 출력
print('y_train 의 크기 :',len(y_train))     # y_train 배열의 크기를 출력
print('x_test 의 크기 :',len(x_test))     # x_test 배열의 크기를 출력
print('y_test 의 크기 :',len(y_test))     # y_test 배열의 크기를 출력

# data 눈으로 확인
plt.imshow(x_train[100], cmap=plt.cm.binary)
plt.imshow(x_train[100]) #뭔차이야
plt.show()

print(y_train[100]) # x_train[1]에 해당하는 라벨값 보여주기.


# index 에 0에서 299 사이 숫자를 지정해 보기
index = 299 # 임의지정 숫자
plt.imshow(x_train[index], cmap=plt.cm.binary)
plt.show()
print((index+1), '번째 이미지의 숫자는 바로 ', y_train[index],'입니다.')
# 인덱스는 시작이 0이니 당연히 +1


print(x_train.shape) # 가위바위보 테스트 자료. 사이즈는 (300, 28, 28, 3)
print(y_train.shape) # 라벨이다보니 그냥 300,
################
################
# 테스트 자료도 채워주기.
################
print(x_test.shape) # 가위바위보 테스트 사이즈는 (70, 28, 28, 3)
print(y_test.shape) # 역시 그냥 70,

# x_train 배열 내부의 최소, 최대값. 픽셀 당 1바이트로 0~255까지 256개 표현가능
print('최소값 : ',np.min(x_train), '최대값 : ',np.max(x_train))

# 최대값으로 나눠주어 정규화. 아니 근데 왜 .0이야
x_train_norm, x_test_norm = x_train /255.0, x_test / 255.0
print('최소값 : ',np.min(x_train_norm), '최대값 : ', np.max(x_train_norm))


#####################################
# 본격적인 가위바위보 설계

model = keras.models.Sequential()
# Conv2D, filter는 커널의 개수,,
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape = (28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32,(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

print('model에 추가된 Layer 개수 : ', len(model.layers))

model.summary()

############################
# 1-4 딥러닝 네트워크 학습시키기

print("Before reshape - x_train_norm shape : {}".format(x_train_norm.shape))
print('Before reshape - x_test_norm shape : {}'.format(x_test_norm.shape))
# 3차원.


######### 이부분은 일단 보류해보자. 3차원이니까

# x_train_reshaped = x_train_norm.reshape(-1, 28, 28, 1)
# x_test_reshaped = x_test_norm.reshape(-1, 28, 28, 1)
# # 데이터 개수에 -1을 쓰면 자동 계산
# # 3차원에서 4차원으로 바꿔줌. 왜??

# print("After Reshape - x_train_reshaped shape : {}".format(x_train_reshaped.shape))
# print("After Reshape - x_test_reshaped shape : {}".format(x_test_reshaped.shape))
# # 4차원
################### 여기까지

# 자료확인
 len(x_train_norm)
 len(y_train)
# 컴파일
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# 학습 시키기.
model.fit(x_train_norm, y_train, epochs=10)

###############################
# 1-5 얼마나 잘 만들었는지 확인하기 - 10회차 정확도 ....100%??

# 실험용 데이터로 점수 확인 - 정확도 77.14%
test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose = 2)

print('test_loss : {}'.format(test_loss))               # 0.0476800762116909
print('test_accuracy : {}'.format(test_accuracy))       # 0.9854999780654907


# 눈으로 확인해보기
predicted_result = model.predict(x_test_norm) # model이 추론한 확률값.
predicted_labels = np.argmax(predicted_result, axis = 1)

idx = 0 # 1번째 x_test 를 살펴보자.
print('model.predict()결과 : ', predicted_result[idx])
print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
print('실제 데이터의 라벨 : ', y_test[idx])

plt.imshow(x_test[idx], cmap=plt.cm.binary)
plt.show()

# model이 추론해 낸 숫자와 실제 라벨의 값이 다른 경우는?
# 눈으로 확인하기
import random
wrong_predict_list = []

for i, _ in enumerate(predicted_labels):
    # i번째 test_labels과 y_test가 다른 경우만 모으기.
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

# wrong_predict_list 에서 랜덤하게 5개만 뽑아봅시다.
samples = random.choices(population=wrong_predict_list, k=5)

for n in samples:
    print("예측확률분포 : " + str(predicted_result[n]))
    print("라벨 : " + str(y_test[n]) + ", 예측결과 : " + str(predicted_labels[n]))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()

# 1차 모델링 끝
##############################


########################################
# 1-6 더 좋은 네트워크 만들기

# 바꿔 볼 수 있는 하이퍼파라미터들

n_channel_1 = 64
n_channel_2 = 128
n_dense = 32
n_train_epoch=100

model=keras.models.Sequential()

model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation = 'relu', input_shape = (28, 28, 3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation= 'relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# 모델 훈련 - 정확도 ...100?
model.fit(x_train_norm, y_train, epochs=n_train_epoch)


# 모델 시험 - 정확도 75.71%

test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose=2)

print('test_loss : {}'.format(test_loss))
print('test_accuracy : {}'.format(test_accuracy))


