# 氨氮显色分析系统 V2 — 单圆片逐个记录

本版本使用 Streamlit Community Cloud，用户只需浏览器即可使用。

## 部署或更新已有网站

1. 打开 GitHub 仓库 https://github.com/hhz-success/Andanshibie
2. 在仓库根目录替换 `app.py` 和 `requirements.txt`（不是上传 ZIP 本身）。
3. 点击 GitHub 的 **Commit changes** 保存。
4. 已部署的 Streamlit 应用通常会自动检测 GitHub 更新并重新部署；若尚未部署，则打开 https://share.streamlit.io/ ，选择该仓库、`main` 分支和 `app.py` 文件，点击 Deploy。

## 使用

- 每次上传一张单圆片照片，检查圆片位置，输入已知浓度（g/L）后保存。
- 在“颜色与浓度拟合”中选择同一实验批次，查看线性拟合和 R²。
- 每次实验结束下载 CSV，下次再导入恢复。云端会话内数据**不是永久存储**。
- 圆片自动定位只是候选位置，必须检查并用滑块调整；绿色圆圈才是实际取色范围。
- 照明、曝光、白平衡和显色时间必须尽量固定；R² 不是准确度或检出限的证明。
