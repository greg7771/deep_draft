import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image

# =========================================================
# 사용법
# =========================================================
# 1) 직접 그린 숫자 이미지를 준비합니다 (사진, 그림판, 태블릿 필기 등 무엇이든 가능).
# 2) 아래 SOURCE_DIR 폴더 안에 이미지를 넣고, 파일 이름 맨 앞 글자를
#    실제 숫자로 지정합니다. 예:
#       0_first.png, 0_second.jpg, 1_myhandwriting.png, 7_test.png ...
#    (맨 앞 문자가 숫자 라벨로 사용됩니다)
# 3) 이 스크립트를 실행하면 custom_dataset.npz 파일이 생성됩니다.
# 4) mnist_cnn.py와 같은 폴더에 custom_dataset.npz를 두면,
#    다음 실행 시 자동으로 감지되어 추가 학습에 사용됩니다.
# =========================================================

SOURCE_DIR = 'my_handwriting'          # 원본 이미지들을 넣어둘 폴더
OUTPUT_PATH = 'custom_dataset.npz'     # 최종 저장 경로
IMG_SIZE = 28
THRESHOLD = 127


def preprocess_image(path):
    # 1. 그레이스케일로 변환
    img = Image.open(path).convert('L')

    # 2. 28x28 크기로 리사이즈 (MNIST와 동일한 입력 크기)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img)

    # 3. MNIST는 배경이 검정(0), 글씨가 흰색(255)인 형태이므로
    #    직접 찍은 사진(배경 흰색, 글씨 검정)이라면 색을 반전시켜야 할 수 있습니다.
    #    필요하면 아래 주석을 해제하세요.
    # arr = 255 - arr

    # 4. 이진화 (0 또는 1)
    binary = (arr > THRESHOLD).astype('uint8')
    return binary


def build_custom_dataset():
    images = []
    labels = []

    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            f"'{SOURCE_DIR}' 폴더가 없습니다. 폴더를 만들고 손글씨 이미지를 넣어주세요."
        )

    for filename in sorted(os.listdir(SOURCE_DIR)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        label_char = filename[0]
        if not label_char.isdigit():
            print(f"건너뜀 (파일명이 숫자로 시작하지 않음): {filename}")
            continue

        label = int(label_char)
        img_path = os.path.join(SOURCE_DIR, filename)

        binary_img = preprocess_image(img_path)
        images.append(binary_img)
        labels.append(label)

        print(f"처리 완료: {filename} -> 라벨 {label}")

    images = np.array(images, dtype='uint8')
    labels = np.array(labels, dtype='int64')

    np.savez(OUTPUT_PATH, images=images, labels=labels)
    print(f"\n총 {len(labels)}개 이미지를 '{OUTPUT_PATH}'에 저장했습니다.")


if __name__ == '__main__':
    build_custom_dataset()
