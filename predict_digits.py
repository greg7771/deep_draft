import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
from tensorflow.keras import models

# mnist_cnn.py 한 번 이상 실행해야함
# MNIST + custom_dateset.npz 학습된 모델
# 예측하려는 손글씨 파일은 to_predict 폴더에 넣으면 됨 (숫자에 맞춰서 입력 안 해도됨)

MODEL_PATH = 'mnist_cnn_model.keras'   # 학습된 모델 파일
PREDICT_DIR = 'to_predict'             # 예측할 손글씨 이미지를 넣는 폴더
IMG_SIZE = 28
THRESHOLD = 127

# 배경 흰색에 검정 글씨
# MNIST는 배경이 검정, 글씨가 흰색 > 반전 필요가능 
# 예측이 이상하게 나오면 이 값을 True로 바꿔보세요.
INVERT_COLOR = False


def preprocess_image(path):
    img = Image.open(path).convert('L')          # 그레이스케일 변환
    img = img.resize((IMG_SIZE, IMG_SIZE))        # 28x28로 리사이즈
    arr = np.array(img)

    if INVERT_COLOR:
        arr = 255 - arr

    binary = (arr > THRESHOLD).astype('float32')  # 0/1 이진화
    return binary.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}'가 없습니다. 먼저 mnist_cnn.py를 실행해 모델을 학습/저장하세요."
        )

    if not os.path.isdir(PREDICT_DIR):
        raise FileNotFoundError(
            f"'{PREDICT_DIR}' 폴더가 없습니다. 폴더를 만들고 예측할 이미지를 넣어주세요."
        )

    print(f"모델을 불러옵니다: {MODEL_PATH}")
    model = models.load_model(MODEL_PATH)

    filenames = [f for f in sorted(os.listdir(PREDICT_DIR))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not filenames:
        print(f"'{PREDICT_DIR}' 폴더에 이미지가 없습니다.")
        return

    print(f"\n총 {len(filenames)}개 이미지를 예측합니다.\n")

    for filename in filenames:
        img_path = os.path.join(PREDICT_DIR, filename)
        img_array = preprocess_image(img_path)

        prediction = model.predict(img_array, verbose=0)
        predicted_label = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        print(f"{filename:30s} -> 예측: {predicted_label}  (확신도 {confidence:.1f}%)")


if __name__ == '__main__':
    main()
