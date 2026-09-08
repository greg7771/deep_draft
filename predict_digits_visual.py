import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import models

# =========================================================
# 사용법
# =========================================================
# 1) mnist_cnn.py를 한 번 이상 실행해서 'mnist_cnn_model.keras'가
#    생성되어 있어야 합니다.
# 2) 예측하고 싶은 손글씨 이미지를 PREDICT_DIR 폴더에 넣습니다.
# 3) 이 스크립트를 실행하면
#    - 각 이미지마다 "원본 이미지 + 숫자별 확률 막대그래프"를 화면에 띄우고
#    - result_visuals 폴더에 결과 이미지를 png로 저장하고
#    - 콘솔에 전체 확률 표를 출력합니다.
# =========================================================

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = 'mnist_cnn_model.keras'
PREDICT_DIR = 'to_predict'
RESULT_DIR = 'result_visuals'
IMG_SIZE = 28
THRESHOLD = 127
INVERT_COLOR = False   # 사진 배경이 흰색/글씨가 검정이면 True로 변경


def preprocess_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)

    if INVERT_COLOR:
        arr = 255 - arr

    binary = (arr > THRESHOLD).astype('float32')
    return binary  # (28, 28), 화면 표시용 원본
    

def show_and_save(filename, binary_img, probabilities):
    predicted_label = int(np.argmax(probabilities))

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 4))

    # 왼쪽: 입력된 손글씨 이미지
    ax_img.imshow(binary_img, cmap='gray')
    ax_img.set_title(f"입력 이미지\n({filename})")
    ax_img.axis('off')

    # 오른쪽: 0~9 각 숫자에 대한 확률 막대그래프
    digits = np.arange(10)
    colors = ['crimson' if d == predicted_label else 'royalblue' for d in digits]
    bars = ax_bar.bar(digits, probabilities, color=colors)

    ax_bar.set_xticks(digits)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel('숫자')
    ax_bar.set_ylabel('확률')
    ax_bar.set_title(f"예측 결과: {predicted_label}  "
                      f"(확신도 {probabilities[predicted_label]*100:.1f}%)")

    # 막대 위에 확률 값 표시
    for bar, prob in zip(bars, probabilities):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{prob:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, f"{os.path.splitext(filename)[0]}_result.png")
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

    print(f"결과 이미지 저장됨: {save_path}")


def print_probability_table(filename, probabilities):
    predicted_label = int(np.argmax(probabilities))
    print(f"\n[{filename}] 예측: {predicted_label}")
    print(f"{'숫자':>4} | {'확률':>8}")
    print("-" * 16)
    for digit, prob in enumerate(probabilities):
        marker = " <-- 예측" if digit == predicted_label else ""
        print(f"{digit:>4} | {prob:8.5f}{marker}")


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

    for filename in filenames:
        img_path = os.path.join(PREDICT_DIR, filename)
        binary_img = preprocess_image(img_path)

        model_input = binary_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        probabilities = model.predict(model_input, verbose=0)[0]  # (10,) 형태

        print_probability_table(filename, probabilities)
        show_and_save(filename, binary_img, probabilities)


if __name__ == '__main__':
    main()
