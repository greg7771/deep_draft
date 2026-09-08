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
#    (학습 데이터: 검은 배경 + 흰 글씨, MNIST 표준 형태)
# 2) 예측하고 싶은 손글씨 이미지를 PREDICT_DIR 폴더에 넣습니다.
#    -> 배경이 검은색이든 흰색이든 상관없이 넣으면 됩니다.
# 3) 이 스크립트가 이미지의 평균 밝기를 분석해서
#    "이 이미지는 배경이 흰색인지 검은색인지"를 자동으로 판단하고,
#    학습 데이터(검은 배경+흰 글씨)와 같은 형태로 자동 반전해줍니다.
# 4) 화면에는 "자동 판단으로 적용된 예측"과
#    "반대로 반전했다면 나왔을 예측" 두 가지를 함께 보여줍니다.
# =========================================================

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = 'mnist_cnn_model.keras'
PREDICT_DIR = 'to_predict'
RESULT_DIR = 'result_visuals'
IMG_SIZE = 28
THRESHOLD = 127

# 이 값을 기준으로 "배경이 흰색에 가깝다"고 판단합니다 (0~255).
# 평균 밝기가 이 값보다 크면 흰 배경으로 보고 자동으로 반전시킵니다.
BRIGHTNESS_CUTOFF = 127


def load_grayscale(path):
    """이미지를 불러와 28x28 그레이스케일 배열로 변환 (아직 이진화 전, 0~255 값)"""
    img = Image.open(path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).astype('float32')


def guess_invert(gray_arr):
    """
    이미지의 평균 밝기를 보고 학습 데이터(검은 배경+흰 글씨)와
    같은 형태로 만들려면 반전이 필요한지 자동으로 판단합니다.

    - 글씨보다 배경이 훨씬 넓은 면적을 차지하므로,
      평균 밝기는 사실상 '배경이 어떤 색인가'를 대변합니다.
    - 평균 밝기가 높다 (밝다) -> 배경이 흰색 -> 학습 데이터와 반대 -> 반전 필요 (True)
    - 평균 밝기가 낮다 (어둡다) -> 배경이 검은색 -> 학습 데이터와 동일 -> 반전 불필요 (False)
    """
    mean_brightness = gray_arr.mean()
    need_invert = mean_brightness > BRIGHTNESS_CUTOFF
    return need_invert, mean_brightness


def binarize(gray_arr, invert):
    arr = (255 - gray_arr) if invert else gray_arr
    return (arr > THRESHOLD).astype('float32')


def predict_probabilities(model, binary_img):
    model_input = binary_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return model.predict(model_input, verbose=0)[0]


def print_probability_table(title, probabilities):
    predicted_label = int(np.argmax(probabilities))
    print(f"\n[{title}] 예측: {predicted_label}")
    print(f"{'숫자':>4} | {'확률':>8}")
    print("-" * 16)
    for digit, prob in enumerate(probabilities):
        marker = " <-- 예측" if digit == predicted_label else ""
        print(f"{digit:>4} | {prob:8.5f}{marker}")


def draw_bar(ax, probabilities, title):
    predicted_label = int(np.argmax(probabilities))
    digits = np.arange(10)
    colors = ['crimson' if d == predicted_label else 'royalblue' for d in digits]
    bars = ax.bar(digits, probabilities, color=colors)

    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    ax.set_xlabel('숫자')
    ax.set_ylabel('확률')
    ax.set_title(title)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{prob:.2f}", ha='center', va='bottom', fontsize=7)


def show_and_save(filename, applied_img, applied_probs, applied_label_txt,
                   alt_img, alt_probs, alt_label_txt):
    fig, (ax_img, ax_applied, ax_alt) = plt.subplots(1, 3, figsize=(13, 4))

    # 왼쪽: 실제로 모델에 입력된(자동 판단 적용된) 이미지
    ax_img.imshow(applied_img, cmap='gray')
    ax_img.set_title(f"입력 이미지\n({filename})\n[{applied_label_txt}]")
    ax_img.axis('off')

    # 가운데: 자동 판단으로 적용된 방향의 예측 확률
    predicted_applied = int(np.argmax(applied_probs))
    draw_bar(ax_applied, applied_probs,
             f"적용된 예측: {predicted_applied}\n"
             f"(확신도 {applied_probs[predicted_applied]*100:.1f}%)")

    # 오른쪽: 반대로 반전했을 경우의 예측 확률 (비교용)
    predicted_alt = int(np.argmax(alt_probs))
    draw_bar(ax_alt, alt_probs,
             f"반전 시 예측: {predicted_alt}\n"
             f"(확신도 {alt_probs[predicted_alt]*100:.1f}%)")

    plt.tight_layout()

    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, f"{os.path.splitext(filename)[0]}_result.png")
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

    print(f"결과 이미지 저장됨: {save_path}")


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
        gray_arr = load_grayscale(img_path)

        need_invert, mean_brightness = guess_invert(gray_arr)

        print(f"\n=== {filename} ===")
        print(f"평균 밝기: {mean_brightness:.1f} "
              f"-> {'흰 배경으로 판단, 반전 적용' if need_invert else '검은 배경으로 판단, 반전 없음'}")

        # 자동 판단으로 '적용된' 방향
        applied_img = binarize(gray_arr, invert=need_invert)
        applied_probs = predict_probabilities(model, applied_img)
        applied_label_txt = "반전 적용됨" if need_invert else "원본 그대로"

        # 비교를 위한 '반대' 방향
        alt_img = binarize(gray_arr, invert=not need_invert)
        alt_probs = predict_probabilities(model, alt_img)
        alt_label_txt = "원본 그대로" if need_invert else "반전 적용됨"

        print_probability_table(f"{filename} / {applied_label_txt} (적용됨)", applied_probs)
        print_probability_table(f"{filename} / {alt_label_txt} (비교용)", alt_probs)

        show_and_save(filename, applied_img, applied_probs, applied_label_txt,
                      alt_img, alt_probs, alt_label_txt)


if __name__ == '__main__':
    main()
