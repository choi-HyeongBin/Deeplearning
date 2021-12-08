# Deeplearning을 활용한 '등산객을 위한 재난 확인'

---

### 프로젝트 설명
> 머신러닝 모델 중에서 이미지를 학습하여 분류하고 새로운 이미지에 대한 분류를 하는 'CNN' 모델을 사용하여 개발하였습니다.

---

### 개발 기간
> 2020년 3학년 2학기 (2개월)
---

### 프로젝트 계획 이유
> 기계학습을 통해 산에서 발생하는 재난 이미지를 학습을 시켜서 재난이 발생 했을 때 빠르게 대처하여 피해를 줄이기 위해 개발하기로 하였습니다.<br>
> 저 포함 3명이 팀을 이루어 개발하였습니다.

---

### 사용 기술
> + 데이터 수집: Kaggle, crawling, Data argumentation
> + 개발 도구: pycharm

---

### 학습 데이터

> 학습데이터로는 총 4가지(평범한 산, 산불이 나는 산, 안개가 가득한 산, 산사태가 일어난 산)로 정하고 kaggle, google crawling 및 Data argumentation 을 통해 각각 3천개의 학습데이터를 만들었습니다. <br> 그 중에서 학습&검증 데이터는 'Train_test_split'을 통해 학습데이터는 75%, 검증 데이터는 25%로 분할 하였고 테스트 데이터는 학습&검증 데이터에 없는 데이터를 따로 받아와서 각 클래스 별로 3개씩 수집하였습니다.
---

### 1차 모델(LeNet-5)
> CNN모델 중 1차모델로 LeNet-5으로 정하여 총 3개의 convolution layer, 2개의 Average pooling layer 및 1개의 fully-connected layer로 구성하여 구현하였습니다.<br>
```
model = Sequential()
    model.add(Conv2D(filters=6, input_shape=X_train.shape[1:], kernel_size=(5, 5), strides=(1, 1), padding='same',
                    activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1, 1), activation='tanh'))
    
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1, 1), activation='tanh'))
   
    model.add(Flatten())

    model.add(Dense(84, activation='tanh'))
  
    model.add(Dense(nb_classes, activation='softmax'))
```
---

### 1차 모델(LeNet-5) 실행 결과
![image](https://user-images.githubusercontent.com/94504100/144980378-d51ce38f-459a-452e-8d85-cb1cfd520ed4.png)
<br>
> 1차 모델 적용 결과 정확도는 0.875가 나왔지만 '평범한 산'을 '안개가 가득한 산'으로 인식하는 문제점이 발견하였습니다.<br>
개선방안으로는 CNN모델(VGG.glexnet)을 적용하고 batch size와 epochs 값을 조정하기로 하였습니다.

---

### 최종 모델(Alexnet)
```
model = Sequential()
    model.add(Conv2D(filters=96, input_shape=X_train.shape[1:], kernel_size=(11, 11), strides=(4, 4), padding='valid',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(Flatten())

    model.add(Dense(4096, input_shape=X_train.shape[1:], activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))
```
> 여러 모델을 적용해 본 결과 Alexnet 모델이 가장 성능이 뛰어나서 Alexnet 모델을 최종 모델로 선정하였으며, 5개의 convolution layer, 3개의 Max pooling layer와 3개의fully-connected layer로 구성하여 개발하였습니다.

---

### 최종 모델(Alexnet) 실행결과
![image](https://user-images.githubusercontent.com/94504100/144982857-09cb5f40-9c89-403c-b0f5-0fa1171a3441.png)
<br>
![image](https://user-images.githubusercontent.com/94504100/144982901-4a0dbbe3-0331-42b0-b78b-4070c6fd4b23.png)

> 최종 모델 정확도는 0.93까지의 높은 정확도를 보여주며 그래프처럼 epochs를 반복할수록 정확도가 높아지는 현상을 볼 수 있었다.


