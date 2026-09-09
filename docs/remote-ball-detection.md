# 篮球检测接入与评估

> 状态基线：2026-08-04。当前默认方案是本地 YOLO，Roboflow 和 Hugging Face
> Hosted Inference 作为可选备用。本地权重、缓存、测试视频和运行产物不提交到 Git。

## 当前实现

工程提供统一的篮球检测接口：

- `local`：Ultralytics YOLO，本地运行，无按次 API 费用
- `roboflow`：Roboflow Hosted API
- `huggingface` / `hf`：Hugging Face Hosted Inference
- `none`：只做姿态分析

分析流程不是简单地在姿态离手帧画一个框，而是在按 FPS 自适应的窗口内检测球，
过滤静止背景球，判断球手接触和飞行轨迹，再对姿态离手帧进行向前或向后修正。

输出的 `release_analysis` 包含：

- `pose_release_frame`
- `contact_frame`
- `separation_frame`
- `final_release_frame`
- `source`
- `ball_candidate_wrist_distance`

## 本地模型

默认模型路径是 `models/basketball_yolo11.pt`，当前评估使用
[Lumos-88/YOLO11-fine-tuned-for-basketball-detection](https://huggingface.co/Lumos-88/YOLO11-fine-tuned-for-basketball-detection)
的 `best.pt`。下载后将文件改名并放到该路径。若默认路径不存在，provider 会回退到
Ultralytics `yolov8n.pt`；首次使用可能联网下载。

```powershell
pip install .[api,local]
basketball-analyzer single --input shooting.mp4 --output artifacts/single_local --ball-provider local
basketball-analyzer ball-eval --provider local --videos shooting.mp4 shoot1.mp4 curry.mp4 --output artifacts/ball_eval_local
```

模型页面标注 MIT，但 YOLO 基础软件的商业使用还受 Ultralytics 许可约束。
因此仓库不分发权重，正式商业化前需完成模型、训练数据和运行时许可证审查。

## 当前验证结论

人工纠正后的 `shooting.mp4` 离手真值约为 161 帧：150 帧球仍在手上，160 帧仍在指尖，
162 帧已经飞出。当前本地流程输出 164（`ball-flight-bound`），比纯姿态结果 172 更接近真值。

篮球微调权重在 `shooting.mp4`、`shoot1.mp4`、`curry.mp4` 的抽样窗口中均有检出，
解决了已评估 Roboflow Hosted 模型在 `shoot1.mp4` 上漏检的问题。但该权重在某些远景会把手
误检为篮球；这类视频可对比 `yolov8n.pt`，最终仍应以时序轨迹而不是单帧置信度决定离手帧。

完整七段视频回归表见 [项目状态与路线图](project-status.md)。这些第三方样例只保存在本机，
来源和再分发边界见 [参考视频说明](reference-videos.md)。

## 远端备用方案

已完成真实联调的候选包括 `basketball-game-detections/9` 和
`basketball-detection-dn6fg/1`。远端 provider 继续保留，便于零本地算力环境试用和横向评估，
但不再作为默认方案，因为它们受 API 配额、网络延迟、服务稳定性和跨场景漏检影响。

```powershell
$env:ROBOFLOW_API_KEY = "your_api_key"
basketball-analyzer single --input shooting.mp4 --output artifacts/single_rf --ball-provider roboflow --roboflow-model-id basketball-game-detections/9
basketball-analyzer ball-eval --provider roboflow --videos shooting.mp4 curry.mp4 --models basketball-game-detections/9 basketball-detection-dn6fg/1 --output artifacts/ball_eval_rf
```

Hugging Face 远端方式使用 `HF_TOKEN` 和 `HF_MODEL_ID`。API 密钥只通过环境变量或系统凭据提供，
不得写入配置文件、文档或 Git 历史。

## 下一步

1. 扩充自有、可授权的视频样本，覆盖机位、遮挡、光照和左右手投篮。
2. 增加球候选的时序连续性和手部误检过滤。
3. 自动切分长视频中的多次投篮，并逐次输出报告。
4. 用标注集统计离手帧 MAE、球检出率和误检率，替代仅靠案例判断模型优劣。
