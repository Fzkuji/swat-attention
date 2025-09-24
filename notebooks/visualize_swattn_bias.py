import numpy as np
import matplotlib.pyplot as plt

# 可视化 swattn bias 曲线

def compute_bias_curve(head_id, num_heads, max_token=120):
    D0 = 1
    D1 = 10 + head_id
    D2 = 80 + 4 * head_id
    start_bias = 1 - head_id / num_heads
    end_bias = -1 + head_id / num_heads
    bias = np.zeros(max_token)
    for d in range(max_token):
        if D0 <= d <= D1:
            slope1 = (end_bias - start_bias) / (D1 - D0)
            bias[d] = start_bias + slope1 * (d - D0)
        elif D1 < d <= D2:
            slope2 = (0 - end_bias) / (D2 - D1)
            bias[d] = end_bias + slope2 * (d - D1)
        else:
            bias[d] = 0
    return bias

if __name__ == "__main__":
    num_heads = 8  # 可调整
    max_token = 120
    plt.figure(figsize=(10, 6))
    for head_id in range(num_heads):
        bias = compute_bias_curve(head_id, num_heads, max_token)
        plt.plot(bias, label=f"head {head_id}")
        # 标注关键点
        D0 = 1
        D1 = 10 + head_id
        D2 = 80 + 4 * head_id
        plt.scatter([D0, D1, D2], [bias[D0], bias[D1], bias[D2]], marker='x')
    plt.title("SWAttn Bias Curve per Head")
    plt.xlabel("Token Index (distance)")
    plt.ylabel("Bias Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

