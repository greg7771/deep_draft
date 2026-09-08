import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import models

# mnist_cnn.py 한 번 이상 실행해야함
# MNIST + custom_dateset.npz 학습된 모델
# 예측하려는 손글씨 파일은 to_predict 폴더에 넣으면 됨 (숫자에 맞춰서 입력 안 해도됨)
# 원본 vs 반전 해서 확률 높은거 최종적으로 보여줌

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = 'mnist_cnn_model.keras'
PREDICT_DIR = 'to_predict' # 예측할 파일 넣을 곳
RESULT_DIR = 'result_visuals' # 예측 결과 png 저장
IMG_SIZE = 28
THRESHOLD = 127


def load_grayscale(path):
    """이미지를 불러와 28x28 1 0 이진화?? 검흰만 남김 (원본 색 그대로, 0~255 값)"""
    img = Image.open(path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).astype('float32')


def binarize(gray_arr, invert):
    """원본 배열은 건드리지 않고, 필요할 때만 반전된 새 배열을 만들어 반환"""
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


def show_and_save(filename, original_img, original_probs, inverted_probs, final_side):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_orig = fig.add_subplot(gs[0, 1])
    ax_inv = fig.add_subplot(gs[0, 2])
    ax_final = fig.add_subplot(gs[1, :])

    # 맨 왼쪽: 원본 이미지 (반전 없이 그대로 보여줌)
    ax_img.imshow(original_img, cmap='gray')
    ax_img.set_title(f"원본 이미지\n({filename})")
    ax_img.axis('off')

    # 가운데: 원본 그대로 넣었을 때 확률
    draw_bar(ax_orig, original_probs, "원본 그대로 예측했을 때 확률")

    # 오른쪽: 반전시켰을 때 확률
    draw_bar(ax_inv, inverted_probs, "반전시켜서 예측했을 때 확률")

    # 아래: 더 확신도 높은 쪽을 최종 예측으로 표시
    final_probs = original_probs if final_side == '원본' else inverted_probs
    final_label = int(np.argmax(final_probs))
    final_conf = final_probs[final_label] * 100
    draw_bar(ax_final, final_probs,
             f"[최종 판단: {final_side} 쪽이 더 확신도 높음]  "
             f"최종 예측 -> {final_label}  (확신도 {final_conf:.1f}%)")

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

        # 원본 파일은 건드리지 않고, gray_arr는 계산용 복사본으로만 사용
        gray_arr = load_grayscale(img_path)

        # 원본 그대로 / 반전 두 버전 모두 계산 (원본 배열은 그대로 두고 새 배열만 생성)
        original_img = binarize(gray_arr, invert=False)
        inverted_img = binarize(gray_arr, invert=True)

        original_probs = predict_probabilities(model, original_img)
        inverted_probs = predict_probabilities(model, inverted_img)

        # 확신도 비교: 각 확률 분포에서 가장 높은 확률(한 숫자에 몰린 정도)이 클수록
        # 더 명확하게 판독했다고 판단
        original_confidence = np.max(original_probs)
        inverted_confidence = np.max(inverted_probs)

        final_side = '원본' if original_confidence >= inverted_confidence else '반전'

        print(f"\n=== {filename} ===")
        print(f"원본 확신도: {original_confidence*100:.1f}%  |  "
              f"반전 확신도: {inverted_confidence*100:.1f}%  "
              f"-> 최종 채택: {final_side}")

        print_probability_table(f"{filename} / 원본 그대로", original_probs)
        print_probability_table(f"{filename} / 반전시킨 경우", inverted_probs)

        show_and_save(filename, original_img, original_probs, inverted_probs, final_side)


if __name__ == '__main__':
    main()