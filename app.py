"""Streamlit ammonia nitrogen single-disc colorimetry recorder."""
from __future__ import annotations

import hashlib
import io
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="氨氮单圆片显色分析", page_icon="🧪", layout="wide")

FEATURES = ["B-R", "B/R", "R", "G", "B", "HSV_H", "HSV_S", "HSV_V", "Lab_L", "Lab_a", "Lab_b"]
NUMERIC = ["浓度_g/L", "R", "G", "B", "B-R", "B/R", "HSV_H", "HSV_S", "HSV_V", "Lab_L", "Lab_a", "Lab_b"]
COLS = ["样品编号", "实验批次", "浓度_g/L", "R", "G", "B", "B-R", "B/R", "HSV_H", "HSV_S", "HSV_V", "Lab_L", "Lab_a", "Lab_b", "文件名", "记录时间"]

if "records" not in st.session_state:
    st.session_state.records = []
if "upload_index" not in st.session_state:
    st.session_state.upload_index = 0
if "notice" not in st.session_state:
    st.session_state.notice = ""


def to_rgb(uploaded):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded.getvalue()))).convert("RGB")
    # Constrain memory / upload variability; sampled colors use central pixel statistics.
    im.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
    return np.asarray(im)


def detect_disc(rgb):
    """Suggest one circular disc; user always verifies and may override."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    sh, sw = gray.shape
    rmin = max(9, int(min(sw, sh) * .025))
    rmax = max(rmin + 2, int(min(sw, sh) * .38))
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=max(25, int(min(sw, sh) * .10)),
                             param1=95, param2=28, minRadius=rmin, maxRadius=rmax)
    if found is None:
        return 50, 50, 18, False
    circles = found[0]
    # Prefer a fairly central circle rather than corners, labels, or edge reflections.
    scores = []
    for x, y, r in circles:
        dist = np.hypot((x - sw / 2) / sw, (y - sh / 2) / sh)
        scores.append(float(r / min(sw, sh)) - .12 * float(dist))
    x, y, r = circles[int(np.argmax(scores))]
    return int(round(x / sw * 100)), int(round(y / sh * 100)), max(2, min(45, int(round(r / min(sw, sh) * 100)))), True


def extract(rgb, x_pct, y_pct, radius_pct):
    h, w = rgb.shape[:2]
    x, y = round(x_pct * (w - 1) / 100), round(y_pct * (h - 1) / 100)
    r = max(3, round(radius_pct * min(w, h) / 100))
    # Analyze only inner 55% radius to avoid meniscus, rim and specular edge.
    inner = max(2, round(.55 * r))
    x0, x1 = max(0, x - inner), min(w, x + inner + 1)
    y0, y1 = max(0, y - inner), min(h, y + inner + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= inner ** 2
    pixels = rgb[y0:y1, x0:x1][mask]
    if len(pixels) < 12:
        raise ValueError("有效像素太少，请调整圆片位置和半径。")
    rgb_median = np.median(pixels, axis=0)
    rr, gg, bb = [float(v) for v in rgb_median]
    one = np.asarray([[rgb_median / 255]], dtype=np.float32)
    hsv = cv2.cvtColor(one, cv2.COLOR_RGB2HSV)[0, 0]
    lab = cv2.cvtColor(one, cv2.COLOR_RGB2LAB)[0, 0]
    result = {
        "R": rr, "G": gg, "B": bb, "B-R": bb - rr, "B/R": bb / max(rr, 1e-6),
        "HSV_H": float(hsv[0]), "HSV_S": float(hsv[1]), "HSV_V": float(hsv[2]),
        "Lab_L": float(lab[0]), "Lab_a": float(lab[1]), "Lab_b": float(lab[2]),
    }
    # Saturation / clipping are warning signs for calibration.
    clipped = float(np.mean(np.any((pixels <= 2) | (pixels >= 253), axis=1)))
    return result, (x, y, r, inner), clipped


def mark(rgb, geometry):
    x, y, r, inner = geometry
    annotated = rgb.copy()
    cv2.circle(annotated, (x, y), r, (240, 80, 30), max(2, r // 70))
    cv2.circle(annotated, (x, y), inner, (10, 180, 80), max(2, r // 80))
    cv2.circle(annotated, (x, y), 3, (255, 20, 20), -1)
    return annotated


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def frame():
    return pd.DataFrame(st.session_state.records, columns=COLS)


st.title("🧪 氨氮显色分析｜单圆片逐个记录")
st.caption("每次只上传一个显色圆片，累积浓度与颜色数据，并实时查看线性关系。浓度单位：g/L。")
st.warning("请保持光源、曝光、白平衡、拍摄距离和显色时间一致。不同照片的 RGB 可能受拍摄条件影响，R² 高不等于定量检测已得到验证。")

if st.session_state.notice:
    st.success(st.session_state.notice)
    st.session_state.notice = ""

n = len(st.session_state.records)
a, b = st.columns(2)
a.metric("已记录样品", n)
b.metric("不同浓度数", len({str(r["浓度_g/L"]) for r in st.session_state.records}))

tab_add, tab_fit, tab_manage = st.tabs(["① 逐个添加圆片", "② 颜色与浓度拟合", "③ 数据保存与恢复"])

with tab_add:
    st.subheader("一次记录一个圆片")
    source = st.radio("图片来源", ["上传图片", "拍照"], horizontal=True)
    if source == "上传图片":
        uploaded = st.file_uploader("上传一张包含单个显色圆片的照片", type=["jpg", "jpeg", "png"], key=f"upload_{st.session_state.upload_index}")
    else:
        uploaded = st.camera_input("拍摄一张圆片照片", key=f"camera_{st.session_state.upload_index}")
    if uploaded is not None:
        try:
            rgb = to_rgb(uploaded)
            digest = hashlib.sha1(uploaded.getvalue()).hexdigest()[:12]
            x_init, y_init, r_init, detected = detect_disc(rgb)
            st.caption("检测到圆片候选区域，请检查。" if detected else "没有可靠检测到圆片，已先选择画面中央；请手动调整。")
            c1, c2 = st.columns([1.2, 1])
            with c2:
                st.markdown("**调整圆片位置（百分比）**")
                x = st.slider("圆心 X（从左到右）", 0, 100, x_init, key=f"x_{digest}")
                y = st.slider("圆心 Y（从上到下）", 0, 100, y_init, key=f"y_{digest}")
                radius = st.slider("圆片半径（占图片短边的百分比）", 2, 45, r_init, key=f"r_{digest}")
            vals, geometry, clipped = extract(rgb, x, y, radius)
            with c1:
                st.image(mark(rgb, geometry), caption="橙色：圆片边界；绿色：实际取色区域", use_container_width=True)
            with c2:
                st.markdown("**本次颜色结果（中位数）**")
                m1, m2, m3 = st.columns(3)
                m1.metric("R", f"{vals['R']:.1f}")
                m2.metric("G", f"{vals['G']:.1f}")
                m3.metric("B", f"{vals['B']:.1f}")
                st.metric("B − R", f"{vals['B-R']:.2f}")
                st.color_picker("圆片中心代表色", value="#%02x%02x%02x" % tuple(int(round(vals[k])) for k in ("R", "G", "B")), disabled=True)
                if clipped > .02:
                    st.warning(f"取色区域约 {clipped:.1%} 的像素接近全黑/全白，可能存在曝光或反光问题。")
            with st.form("save_disc"):
                label = st.text_input("样品编号", value=f"S{len(st.session_state.records) + 1:03d}")
                batch = st.text_input("实验批次（同一组拍摄条件填写相同名称）", value="实验1")
                conc = st.number_input("已知氨氮浓度（g/L）", min_value=0.0, value=0.0, step=0.01, format="%.5f")
                submitted = st.form_submit_button("➕ 保存此圆片，继续下一个", type="primary", use_container_width=True)
                if submitted:
                    if not label.strip() or not batch.strip():
                        st.error("请填写样品编号和实验批次。")
                    else:
                        row = {"样品编号": label.strip(), "实验批次": batch.strip(), "浓度_g/L": float(conc), **vals,
                               "文件名": uploaded.name, "记录时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        st.session_state.records.append(row)
                        st.session_state.upload_index += 1
                        st.session_state.notice = f"已保存 {label.strip()}（{conc:g} g/L）。可以继续上传下一个圆片。"
                        st.rerun()
        except Exception as exc:
            st.error(f"图片处理失败：{exc}")
    else:
        st.info("上传照片后，先确认圆片取色区域，再输入已知浓度并保存。每次只显示当前圆片照片。")

with tab_fit:
    df = frame()
    if df.empty:
        st.info("先在第一个标签页记录样品。至少需要 3 个数据点且有不同浓度才能做有意义的线性拟合。")
    else:
        batches = sorted(df["实验批次"].dropna().astype(str).unique().tolist())
        batch = st.selectbox("选择实验批次（避免混合不同光照条件）", batches)
        selected = df[df["实验批次"].astype(str) == batch].copy()
        feature = st.selectbox("颜色指标", FEATURES)
        st.caption("可以分别比较 B−R、RGB、HSV 和 Lab。不同指标的 R² 需要结合残差、重复性和独立验证共同判断。")
        lo, hi = float(selected["浓度_g/L"].min()), float(selected["浓度_g/L"].max())
        if hi > lo:
            bounds = st.slider("选择用于拟合的浓度范围（g/L）", min_value=lo, max_value=hi,
                               value=(lo, hi), step=max((hi-lo)/1000, 0.00001), format="%.5f")
            selected = selected[selected["浓度_g/L"].between(*bounds)]
        fit_by_mean = st.checkbox("同浓度重复样品先求均值，再拟合", value=False)
        st.dataframe(selected[["样品编号", "浓度_g/L", "R", "G", "B", "B-R", "B/R", "Lab_b"]], hide_index=True, use_container_width=True)
        d = selected[["浓度_g/L", feature]].apply(pd.to_numeric, errors="coerce").dropna()
        if fit_by_mean:
            d = d.groupby("浓度_g/L", as_index=False)[feature].mean()
        x, y = d["浓度_g/L"].to_numpy(float), d[feature].to_numpy(float)
        if len(x) >= 3 and len(np.unique(x)) >= 3 and np.ptp(y) > 1e-10:
            slope, intercept = np.polyfit(x, y, 1)
            prediction = slope*x + intercept
            r2 = 1 - float(np.sum((y-prediction)**2) / np.sum((y-y.mean())**2))
            c1, c2, c3 = st.columns(3)
            c1.metric("斜率", f"{slope:.5g}")
            c2.metric("截距", f"{intercept:.5g}")
            c3.metric("R²", f"{r2:.4f}")
            st.write(f"**线性方程：** {feature} = {slope:.6g} × C {'+' if intercept >= 0 else '−'} {abs(intercept):.6g}，其中 C 单位为 g/L。")
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            ax.scatter(x, y, s=55, label="Samples")
            line_x = np.linspace(x.min(), x.max(), 200)
            ax.plot(line_x, slope*line_x+intercept, label="Linear fit")
            ax.set_xlabel("Ammonia nitrogen concentration (g/L)")
            ax.set_ylabel(feature)
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button("下载当前拟合图 PNG", buf.getvalue(), file_name="calibration_plot.png", mime="image/png")
            plt.close(fig)
            st.caption("说明：改变浓度范围可能提高 R²，但不能仅凭挑选区间宣称方法线性。请记录完整范围，并用独立重复样品验证。")
        else:
            st.info("当前筛选范围内至少需要 3 个不同浓度点，且颜色指标不能完全相同，才能进行线性拟合。")

with tab_manage:
    df = frame()
    st.subheader("全部记录")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ 下载全部记录 CSV（建议每次实验结束都保存）", csv_bytes(df),
                       file_name="ammonia_disc_records.csv", mime="text/csv", disabled=df.empty)
    st.caption("注意：云端免费版默认只在当前浏览器会话中保留记录。关闭页面、应用重启或会话失效可能丢失数据。请下载 CSV，下次再导入继续分析。")
    st.divider()
    st.markdown("**从之前保存的 CSV 恢复记录**")
    old_file = st.file_uploader("选择本程序导出的 CSV 文件", type="csv", key="restore_csv")
    if old_file is not None and st.button("导入 CSV（与当前数据合并）"):
        try:
            old = pd.read_csv(old_file, encoding="utf-8-sig")
            missing = [c for c in COLS if c not in old.columns]
            if missing:
                st.error("CSV 缺少必要字段：" + "、".join(missing))
            else:
                for col in NUMERIC:
                    old[col] = pd.to_numeric(old[col], errors="coerce")
                old = old.dropna(subset=["浓度_g/L", "R", "G", "B"])
                combined = pd.concat([df, old[COLS]], ignore_index=True)
                # Avoid duplicate import on repeated clicks.
                combined = combined.drop_duplicates(subset=["样品编号", "实验批次", "文件名", "记录时间"], keep="first")
                st.session_state.records = combined.to_dict("records")
                st.session_state.notice = f"已导入并合并记录，现在共 {len(combined)} 条。"
                st.rerun()
        except Exception as exc:
            st.error(f"CSV 读取失败：{exc}")
    st.divider()
    if st.session_state.records and st.button("撤销最后一条记录"):
        st.session_state.records.pop()
        st.rerun()
    if st.checkbox("我已备份 CSV，确认要清空当前全部记录"):
        if st.button("清空全部记录", type="secondary"):
            st.session_state.records = []
            st.session_state.notice = "当前记录已清空。"
            st.rerun()
