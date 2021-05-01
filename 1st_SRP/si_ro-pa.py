import  tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, glob

#############################################################################
# 사진 전처리 ##################################################################
#############################################################################

target = 'papers'
#target = 'rocks'
#target = 'scissors'

images = glob.glob('' + '/*.jpg')
#해당 폴더 내의 모든 jpg파일을 불러서 image에 저장하기.
target_size = (28, 28)
for img in images:
    old_img = Image.open(img)
    new_img = old_img.resize(target_size, image.ANTIALIAS)
    new_img.save(img, 'JPEG')
print('{} images resize Complete!'.format(target))


#############################################################################
# 사진파일 불러오기 ##############################################################
#############################################################################
# loading_train_data
#############################################################################
        # 가위 : 2, 바위 : 0, 보 : 5
def load_train_data(img_path):
    number_of_data = 300
    img_size = 28
    color = 3

    imgs = np.zeros(number_of_data * img_size * img_size * color,
           dtype = np.int32).reshape(number_of_data, img_size, img_size, color)
    # 이미지 데이터와 라벨을 담을 행렬 영역 생성 - 숫자 * 크기 * 크기 * 색
    labels = np.zeros(number_of_data, dtype = np.int32)

    idx = 0
    idx_number_count = 0
    SRP_lists = ['scissors', 'rocks', 'papers']
    for SRP in SRP_lists:
        for file in glob.iglob(img_path+'/'+SRP+'/*.jpg'):
            img = np.array(Image.open(file), dtype = np.int32)
            imgs[idx,:,:,:] = img # 데이터 영역에 이미지 행렬을 복사
            if SRP == 'scissors':
                labels[idx] = 2
            elif SRP == 'rocks':
                labels[idx] = 0
            else:
                labels[idx] = 5
            idx = idx + 1

    print('학습 데이터(x_train)의 이미지 개수는',idx,'입니다.')
    return imgs, labels


def load_test_data(img_path):
    number_of_data = 70
    img_size = 28
    color = 3

    imgs = np.zeros(number_of_data * img_size * img_size * color,
           dtype = np.int32).reshape(number_of_data, img_size, img_size, color)
    # 이미지 데이터와 라벨을 담을 행렬 영역 생성 - 숫자 * 크기 * 크기 * 색
    labels = np.zeros(number_of_data, dtype = np.int32)

    idx = 0
    idx_number_count = 0
    SRP_lists = ['scissors', 'rocks', 'papers']
    for SRP in SRP_lists:
        for file in glob.iglob(img_path+'/'+SRP+'/*.jpg'):
            img = np.array(Image.open(file), dtype = np.int32)
            imgs[idx,:,:,:] = img # 데이터 영역에 이미지 행렬을 복사
            if SRP == 'scissors':
                labels[idx] = 2
            elif SRP == 'rocks':
                labels[idx] = 0
            else:
                labels[idx] = 5
            idx = idx + 1

    print('학습 데이터(x_train)의 이미지 개수는',idx,'입니다.')
    return imgs, labels


# 데이터 불러올 주소 설정
train_image_dir_path = '/Users/heechankang/projects/pythonworkspace/exploratory/samples'
test_image_dir_path = '/Users/heechankang/projects/pythonworkspace/exploratory/test_data'

# 불러온 주소를 가지고 load_data 함수 실행, 데이터 가져오기
x_train, y_train = load_train_data(train_image_dir_path)
x_test, y_test = load_test_data(test_image_dir_path)

#############################################################################
# 불러온 데이터를 처리 ###########################################################
#############################################################################

# 데이터 잘 불러와 졌는지, 크기 확인
print('x_train의 크기 :', len(x_train))         # x_train 배열의 크기를 출력
print('y_train의 크기 :', len(y_train))         # y_train 배열의 크기를 출력
print('x_test의 크기 :', len(x_test))           # x_test 배열의 크기를 출력
print('y_test의 크기 :', len(y_test))           # y_test 배열의 크기를 출력

# 동, 데이터 형태도 확인
print(x_train.shape)    # 테스트자료 크기 출력
print(y_train.shape)    # 테스트자료 크기 출력
print(x_test.shape)     # 테스트자료 크기 출력
print(y_test.shape)     # 테스트자료 크기 출력

# 데이터 이미지 눈으로 확인
check_index = 50 # 임의지정 숫자
plt.imshow(x_train[check_index], cmap = plt.cm.binary)
plt.show()
print((check_index+1), '번째 이미지의 숫자는 바로', y_train[check_index], '입니다.')
                # 인덱스는 사림이 보는 숫자니까 +1

# 배열 내부의 최소, 최대값 확인. 픽셀당 1바이트 0~255니까.
print('최소값 : ', np.min(x_train), '최대값 : ', np.max(x_train))
print('최소값 : ', np.min(y_train), '최대값 : ', np.max(y_train))
print('최소값 : ', np.min(x_test), '최대값 : ', np.max(x_test))
print('최소값 : ', np.min(y_test), '최대값 : ', np.max(y_test))

# 정규화 해주기.
x_train_norm, x_test_norm = x_train / 255, x_test / 255

# 정규화 결과 확인 이상 무
print('최소값 : ', np.min(x_train_norm), '최대값 : ', np.max(x_train_norm))
print('최소값 : ', np.min(x_test_norm), '최대값 : ', np.max(x_test_norm))

#############################################################################
# 본격적인 모델 만들기 ###########################################################
#############################################################################

n_channel_1 = 64
n_channel_2 = 128
n_dense = 32
n_train_epoch=100

model=keras.models.Sequential()
# Conv2D, filter는 커널의 개수,,
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation = 'relu', input_shape = (28, 28, 3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation= 'relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

print('model에 추가된 Layer 개수 : ', len(model.layers))
model.summary()
# 모델 요약정리 / 레이어 수 등 정보확인

# 모델 컴파일
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#############################################################################
# 네트워크 학습시키기 ############################################################
#############################################################################

model.fit(x_train_norm, y_train, epochs=n_train_epoch, verbose = 1)
          # 트레이닝 데이터,     답,            횟수,         표기방법 선정

#############################################################################
# 실험용 데이터로 학습결과 확인 ####################################################
#############################################################################

test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose = 2)
          # 1차 결과는 77.14%
          # 2차 epochs는 100으로 설정해서 재시도, 75.71%,, 왜 더 떨어지냐,,

print('test_loss : {}'.format(test_loss))
        # 1차 결과 : 1.698853850364685
print('test_accuracy : {}'.format(test_accuracy))
        # 2차 결과 : 0.7714285850524902

#############################################################################
# 눈으로 확인해보기
predicted_result = model.predict(x_test_norm) # model이 추론한 확률값.
predicted_labels = np.argmax(predicted_result, axis = 1)

idx = 0 # 1번째 x_test 를 살펴보자.
print('model.predict()결과 : ', predicted_result[idx])
print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
print('실제 데이터의 라벨 : ', y_test[idx])
plt.imshow(x_test[idx], cmap=plt.cm.binary)
plt.show()

#############################################################################
# 틀린 경우 확인하기
import random
wrong_predict_list = []

for i, _ in enumerate(predicted_labels):
    # 이건 신기하다. enumerate 함수로 라벨의 번호만 뽑는건가. 근데 좀 이상한데
    # i번째 test_labels과 y_test가 다른 경우만 모으기.
    if predicted_labels[i] != y_test[i]:
        # 왜 뽑아놓고 또 인덱스로 불러와? 라벨을 그냥 쓰면 되는거 아닌가?
        wrong_predict_list.append(i)

samples = random.choices(population = wrong_predict_list, k=5)
# wrong_predict_list 에서 랜덤하게 5개 뽑아서 samples에 저장

for n in samples:
    print('예측확률분포 : ' + str(predicted_result[n]))
    print('라벨 : ' + str(y_test[n]) + ', 예측결과 : ' + str(predicted_labels[n]))
    plt.imshow(x_test[n], cmap = plt.cm.binary)
    plt.show()


#########################################################
# 싹다 열어보기
for n in wrong_predict_list:
    print('예측확률분포 : ' + str(predicted_result[n]))
    print('라벨 : ' + str(y_test[n]) + ', 예측결과 : ' + str(predicted_labels[n]))
    plt.imshow(x_test[n], cmap = plt.cm.binary)
    plt.show()
