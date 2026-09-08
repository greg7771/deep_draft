import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, datasets

# =========================================================
# 경로 설정
# =========================================================
MODEL_PATH = 'mnist_cnn_model.keras'      # 학습된 모델이 저장될 파일
CUSTOM_DATA_PATH = 'custom_dataset.npz'   # 직접 만든 손글씨 데이터셋 파일
                                           # (이 파일이 있으면 자동으로 추가 학습)


# =========================================================
# 1. MNIST 데이터 불러오기 + 전처리 (0/1 이진화)
# =========================================================
def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    threshold = 127
    train_images = (train_images > threshold).astype('float32').reshape(-1, 28, 28, 1)
    test_images = (test_images > threshold).astype('float32').reshape(-1, 28, 28, 1)

    return train_images, train_labels, test_images, test_labels


# =========================================================
# 2. 직접 만든 데이터셋 불러오기
#    (custom_dataset.npz 안에 images, labels 배열이 있어야 함)
# =========================================================
def load_custom_dataset(path):
    data = np.load(path)
    images = data['images']  # (N, 28, 28), 값은 0 또는 1
    labels = data['labels']  # (N,), 정수 라벨 0~9

    images = images.astype('float32').reshape(-1, 28, 28, 1)
    return images, labels


# =========================================================
# 3. 모델 구조 정의
# =========================================================
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# =========================================================
# 4. 메인 로직
#    - 저장된 모델이 있고 새 데이터가 없으면 -> 불러오기만 하고 학습 생략
#    - 저장된 모델이 없으면 -> 새로 만들어서 학습
#    - custom_dataset.npz가 있으면 -> 항상 그 데이터를 합쳐서 재학습
# =========================================================
train_images, train_labels, test_images, test_labels = load_mnist()

need_training = False

if os.path.exists(MODEL_PATH):
    print(f"저장된 모델을 불러옵니다: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)
else:
    print("저장된 모델이 없어 새로 생성합니다.")
    model = build_model()
    need_training = True

if os.path.exists(CUSTOM_DATA_PATH):
    print(f"직접 만든 데이터셋을 발견했습니다: {CUSTOM_DATA_PATH} -> 함께 학습합니다.")
    custom_images, custom_labels = load_custom_dataset(CUSTOM_DATA_PATH)
    train_images = np.concatenate([train_images, custom_images], axis=0)
    train_labels = np.concatenate([train_labels, custom_labels], axis=0)
    need_training = True

if need_training:
    model.fit(train_images, train_labels, epochs=5)
    model.save(MODEL_PATH)
    print(f"학습 완료, 모델을 저장했습니다: {MODEL_PATH}")
else:
    print("새로운 데이터가 없어 학습을 건너뜁니다.")

# =========================================================
# 5. 평가
# =========================================================
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
