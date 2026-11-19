import io
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="氨氮显色分析系统", page_icon="🧪", layout="wide")

st.title("🧪 氨氮显色图像分析系统")
st.caption("NH₃-N Colorimetric Image Analysis · 在线版")

st.info(
    "使用提示：尽量在固定光源、固定曝光和白平衡下拍照。自动识别结果必须先目视核对；"
    "图像颜色只能用于建立经验校准关系，不能替代标准方法验证。"
)

with st.sidebar:
    st.header("识别设置")
    min_radius = st.slider("圆片最小半径（像素）", 8, 100, 20)
    max_radius = st.slider("圆片最大半径（像素）", 15, 180, 75)
    param2 = st.slider("圆片检测灵敏度（越低越容易检出）", 10, 80, 28)
    blur_size = st.select_slider("图像平滑程度", options=[3, 5, 7, 9, 11], value=5)
    sort_mode = st.radio("圆片排序方式", ["从左到右（同一行）", "从上到下（同一列）", "按行排列（先上后下）"])

uploaded = st.file_uploader(
    "上传显色反应照片（可一次上传多张）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

def detect_circles(rgb, min_r, max_r, p2, blur):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, blur)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(18, min_r * 1.5),
        param1=90, param2=p2, minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return []
    return [(int(round(x)), int(round(y)), int(round(r))) for x, y, r in circles[0]]

def order_circles(circles, mode):
    if mode == "从左到右（同一行）":
        return sorted(circles, key=lambda c: c[0])
    if mode == "从上到下（同一列）":
        return sorted(circles, key=lambda c: c[1])
    # 按圆心纵坐标分行，再按横坐标排序；阈值按半径自适应
    if not circles:
        return []
    avg_r = np.median([c[2] for c in circles])
    remaining = sorted(circles, key=lambda c: c[1])
    rows = []
    while remaining:
        anchor_y = remaining[0][1]
        row = [c for c in remaining if abs(c[1] - anchor_y) <= max(18, avg_r * 1.2)]
        remaining = [c for c in remaining if c not in row]
        rows.append(sorted(row, key=lambda c: c[0]))
    return [c for row in rows for c in row]

def sample_features(rgb, circles):
    h, w = rgb.shape[:2]
    rows = []
    for idx, (x, y, r) in enumerate(circles, 1):
        # 只取圆片内部区域，避开边缘反光；半径取 55%
        rr = max(2, int(r * 0.55))
        x0, x1 = max(0, x-rr), min(w, x+rr+1)
        y0, y1 = max(0, y-rr), min(h, y+rr+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx-x)**2 + (yy-y)**2 <= rr**2
        crop = rgb[y0:y1, x0:x1]
        pix = crop[mask]
        if len(pix) == 0:
            continue
        med = np.median(pix, axis=0)
        r8, g8, b8 = [float(v) for v in med]
        lab = cv2.cvtColor(np.uint8([[med]]), cv2.COLOR_RGB2LAB)[0, 0]
        rows.append({
            "识别序号": idx, "圆心x": x, "圆心y": y, "半径": r,
            "R": r8, "G": g8, "B": b8,
            "B-R": b8-r8, "B/R": b8/(r8+1e-9),
            "Lab_L": float(lab[0]), "Lab_a": float(lab[1])-128,
            "Lab_b": float(lab[2])-128,
        })
    return pd.DataFrame(rows)

def fit_plot(df, feature, conc_col):
    d = df[[conc_col, feature]].dropna().sort_values(conc_col)
    x = d[conc_col].to_numpy(float)
    y = d[feature].to_numpy(float)
    if len(d) < 3 or len(np.unique(x)) < 2:
        return None, None, None
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope*x + intercept
    ss_res = np.sum((y-pred)**2)
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2

if uploaded:
    all_results = []
    for file_index, file in enumerate(uploaded):
        st.divider()
        st.subheader(f"照片 {file_index+1}：{file.name}")
        image = Image.open(file).convert("RGB")
        rgb = np.array(image)
        circles = order_circles(
            detect_circles(rgb, min_radius, max_radius, param2, blur_size),
            sort_mode
        )

        if not circles:
            st.warning("没有自动识别到圆片。请调整左侧半径范围或检测灵敏度后重试。")
            st.image(rgb, caption="原始照片", use_container_width=True)
            continue

        preview = rgb.copy()
        for i, (x, y, r) in enumerate(circles, 1):
            cv2.circle(preview, (x, y), r, (255, 60, 30), 2)
            cv2.circle(preview, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(preview, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        left, right = st.columns([1, 1])
        with left:
            st.image(preview, caption=f"自动识别结果：{len(circles)} 个圆片（请核对编号和位置）", use_container_width=True)
        with right:
            st.write("按图中编号顺序输入对应的氨氮浓度，单位 g/L。")
            default_conc = ", ".join(["0"] * len(circles))
            raw = st.text_input(
                f"浓度列表（照片 {file_index+1}）",
                value=default_conc,
                key=f"conc_{file_index}",
                help="例如：0, 2, 5, 10, 20, 30。数量必须与识别到的圆片数一致。"
            )
            try:
                concs = [float(x.strip()) for x in re.split(r"[,，;\s]+", raw.strip()) if x.strip()]
            except ValueError:
                concs = []
            if len(concs) != len(circles):
                st.error(f"目前识别到 {len(circles)} 个圆片，但输入了 {len(concs)} 个浓度。请调整到数量一致。")
                continue
            feats = sample_features(rgb, circles)
            feats.insert(1, "氨氮浓度_g/L", concs[:len(feats)])
            feats.insert(0, "照片", file.name)
            st.dataframe(feats, use_container_width=True, hide_index=True)
            all_results.append(feats)

    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        st.divider()
        st.header("颜色—浓度关系分析")
        st.caption("若多张照片的照明、曝光或白平衡不同，不应直接混合拟合；建议先分别分析，或加入统一色卡进行校正。")
        feature = st.selectbox("选择颜色指标", ["B-R", "R", "G", "B", "B/R", "Lab_L", "Lab_a", "Lab_b"])
        slope, intercept, r2 = fit_plot(results, feature, "氨氮浓度_g/L")
        if slope is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("斜率", f"{slope:.6g}")
            c2.metric("截距", f"{intercept:.6g}")
            c3.metric("R²", f"{r2:.5f}")
            st.latex(rf"{feature} = {slope:.6g}C {'+' if intercept >= 0 else '-'} {abs(intercept):.6g}")
            fig, ax = plt.subplots()
            ax.scatter(results["氨氮浓度_g/L"], results[feature], label="样品")
            x = np.linspace(results["氨氮浓度_g/L"].min(), results["氨氮浓度_g/L"].max(), 200)
            ax.plot(x, slope*x+intercept, label="线性拟合")
            ax.set_xlabel("氨氮浓度 (g/L)")
            ax.set_ylabel(feature)
            ax.set_title(f"{feature} 与氨氮浓度的线性关系")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("至少需要 3 个样品点，且浓度不能全部相同，才能进行线性拟合。")
        st.download_button(
            "下载颜色数据 CSV",
            data=results.to_csv(index=False).encode("utf-8-sig"),
            file_name="ammonia_color_data.csv",
            mime="text/csv"
        )
else:
    st.markdown("### 使用步骤")
    st.markdown(
        "1. 上传包含显色圆片的照片。  \n"
        "2. 调整圆片检测参数，确认红色圆圈准确覆盖样品。  \n"
        "3. 按识别编号输入每个圆片对应的标准浓度。  \n"
        "4. 选择颜色指标，查看拟合方程和 R²，并下载 CSV 数据。"
    )
    st.warning("当前版本先提供自动圆片检测与人工核对流程；若照片背景复杂，可能需要调整检测参数。")
