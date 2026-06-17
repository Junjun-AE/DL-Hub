# Deep_learning_tools4_final_v2 (DL-Hub 训练平台)

GUI 训练/转换/打包平台, 6 大任务模型训练 + 导出为 .dlhub/.pkg 给推理 toolkit。

## v15.1 本轮改动
- model_conversion/dlhub_packager.py    sevseg 写 approx_seg 近似分割元信息
- model_conversion/yolo_export_worker.py sevseg 特征导出分支 (export_scoreyolo_onnx)
- model_image_sevseg/app.py             修复 MaskGeneratorV2→V3 标注
- 9 个 GUI 文件 (各 app.py + ocr/patchcore 的 panel) 加 logger.debug 留痕

## 注意
- 已剔除 dlhub_project/dlhub/frontend/node_modules (90M, 前端依赖)。
  恢复前端运行: cd dlhub_project/dlhub/frontend && npm install
- frontend/src 与 frontend/dist (源码+构建产物) 已保留。
- ultralytics 依赖已包含 (model_image_sevseg/ultralytics)。

## 启动
python run_dlhub.py  (或 dlhub_project/run_dlhub.py)
