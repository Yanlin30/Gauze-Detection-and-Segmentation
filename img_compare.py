import cv2
import numpy as np
from typing import List, Dict
from scipy import stats

# ========== 预处理与指标 ==========
def preprocess(gray, denoise=True, clahe=True):
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe_op.apply(gray)
    return gray

def laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())

def tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    g2 = gx*gx + gy*gy
    return float(np.mean(g2))

def combined_score(img_bgr: np.ndarray, denoise=True, clahe=True, sobel_ksize=3,
                   w_lap=0.5, w_ten=0.5) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray, denoise=denoise, clahe=clahe)
    lap = laplacian_var(gray)
    ten = tenengrad(gray, ksize=sobel_ksize)

    # 简单加权融合（可调权重）
    comb = w_lap * lap + w_ten * ten
    return {"laplacian": lap, "tenengrad": ten, "combined": comb}

# ========== txt读取 ==========
def read_txt_to_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

# ========== 分析组间差异 ==========
def analyze_groups(group1_paths: List[str], group2_paths: List[str], metric="combined") -> Dict:
    scores1, scores2 = [], []
    for p in group1_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            s = combined_score(img)
            scores1.append(s[metric])
            if "V16_11" in p:
                print(s)

    for p in group2_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            s = combined_score(img)
            scores2.append(s[metric])
            if "V14_11" in p:
                print(s)
    scores, scores2 = np.array(scores1), np.array(scores2)
    

    print(len(scores1), len(scores2))
    all_scores = np.concatenate([scores1, scores2])  # 合并两组
    min_val, max_val = all_scores.min(), all_scores.max()
    print(min_val, max_val)
    if max_val > min_val:  # 避免除零
        all_scores = (all_scores - min_val) / (max_val - min_val)
    else:
        all_scores = np.zeros_like(all_scores)

    # 重新拆分回两组
    scores1 = all_scores[:len(scores1)]
    scores2 = all_scores[len(scores1):]
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)

    # Welch’s t 检验
    t_stat, p_val = stats.ttest_ind(scores1, scores2, equal_var=False)

    result = {
        "metric": metric,
        "group1": {"n": len(scores1), "mean": mean1, "var": var1},
        "group2": {"n": len(scores2), "mean": mean2, "var": var2},
        "t_stat": t_stat,
        "p_value": p_val
    }
    return result