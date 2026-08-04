# Basketball Analyzer

一个面向“投篮辅助 APP”方向演进的篮球投篮分析项目。当前仓库已经从实验脚本整理成了可维护的工程骨架，支持单视频分析、球星动作对比、API 调用，以及可插拔的远端篮球检测接入。

## 项目状态

当前阶段：`MVP 工程版`

已经完成：
- 将原始实验脚本重构为 `src/` 下的工程化 Python 包
- 建立统一服务层、CLI 和 FastAPI API
- 跑通真实视频单次分析和球星对比分析
- 固定 `mediapipe==0.10.14`，解决当前 Windows 环境兼容问题
- 输出标准化 JSON、关键帧截图、骨架叠加视频
- 接入本地免费篮球检测（Ultralytics YOLO，默认篮球微调模型 `models/basketball_yolo11.pt`），远端 `Roboflow` / `Hugging Face` 保留为可选
- 增加篮球检测批量评估入口，支持多视频、多模型对比（含 local provider）
- 本地篮球微调模型在 shooting / shoot1 / curry 上检测率均 100%（Roboflow Hosted 在 shoot1 上为 0%）
- 输出视频已支持篮球检测框可视化
- `release_analysis` 已支持姿态离手帧、最后贴手帧、最终离手帧、球手分离帧和分离趋势联合输出
- 已将离手判定从“看到球就确认”升级为“最后贴手帧 -> 下一帧离手”的修正规则
- 已支持球轨迹分类（静止背景球抑制 / 飞行证据）与 `ball-flight-bound` 回溯
- 已支持主体人物锁定（时序连续性过滤）与远景小人物自适应过滤
- 主手判定升级为多信号融合（腕速度 + 举高幅度 + 可见度）
- FastAPI 已提供内嵌网页上传页与 `/tasks/*` 异步任务队列

当前还没完成：
- 多次连续投篮长视频的自动切片
- 篮球微调模型在全身远景场景会把手误检为球（该类视频建议 `--local-model-path yolov8n.pt`）
- 任务队列持久化、批量任务

## 当前能力

- 单视频投篮分析
  - 自动提取人体姿态关键点
  - 自动判断主手
  - 自动截取投篮片段
  - 自动定位姿态离手帧
  - 结合篮球检测回溯“最后贴手帧”和“最终离手帧”
  - 计算基础动作指标并生成中文建议
  - 输出带骨架、篮球框和说明面板的分析视频
- 双视频动作对比
  - 对齐双方离手时刻
  - 将参考球员骨架映射到用户身体尺度
  - 输出双骨架叠加对比视频
- 产品化交付入口
  - CLI
  - FastAPI 服务
  - 文档与测试骨架
- 可插拔篮球检测
  - 默认本地免费模型（Ultralytics YOLO 篮球微调权重）
  - 可选 Roboflow Hosted API / Hugging Face Hosted Inference
  - 支持低成本抽帧检测
  - 支持批量评估多个模型并输出汇总 JSON
- 前端与任务队列
  - 内嵌网页上传页（`GET /`）
  - `/tasks/single`、`/tasks/compare` 异步任务提交与状态查询

## 已验证结果

当前仓库已经在本地跑通这些流程：
- `shooting.mp4` 单视频分析
- `shooting.mp4` 对比 `curry.mp4`
- FastAPI `/health`
- FastAPI `/analyze/single`
- FastAPI `/analyze/compare`
- FastAPI `/tasks/*` 异步任务与网页上传页
- 本地免费篮球检测在 7 个真实视频上的回归（见 `docs/project-status.md`）

这次最新验证的关键结果（状态基线：2026-08-04，纠正旧结论）：
- `shooting.mp4` 逐帧人工核验：150 帧球仍在手上，真实离手 ≈ `161` 帧（旧文档“150 为离手帧”有误）
- 当前算法最终离手帧为 `164`（ball-flight-bound，误差 +3），纯姿态启发式为 `172`（误差 +11）
- `shoot1.mp4`（旧 Hosted 模型全漏检）现由本地模型 100% 检出，离手帧 `109` 获 pose+ball 双重确认
- 远景切片 clip3 覆盖“球贴手贯穿 pose 帧”场景，离手帧由 65 正确推进到 `70`

完整回归表和当前限制见 [项目状态与路线图](docs/project-status.md)。`artifacts/` 为本地运行产物，
不随仓库提交，避免把旧产物误当成当前算法结论。

## 工程框架

```text
basketball-master/
├─ src/basketball_analyzer/
│  ├─ analysis/           # 姿态启发式分析、对比映射、指标计算
│  ├─ detection/          # 远端篮球检测提供方与抽帧调用逻辑
│  ├─ rendering/          # 骨架叠加、篮球框渲染、对比视频、关键帧输出
│  ├─ api.py              # FastAPI 应用
│  ├─ cli.py              # 命令行入口
│  ├─ config.py           # 参数配置
│  ├─ models.py           # 领域模型与标准化输出对象
│  ├─ pose.py             # MediaPipe Pose 提取
│  └─ service.py          # 对外统一服务层
├─ docs/                  # 状态、工程说明、远端检测方案
├─ tests/                 # 基础单元测试
├─ overlay_pose_video.py  # 原实验脚本，保留做参考
├─ my_vs_curry_overlay.py # 原实验脚本，保留做参考
└─ basketball-Gemini.py   # 原展示脚本，保留做参考
```

## 当前技术选型

- 姿态估计：`MediaPipe Pose`
- 数值分析：`NumPy`
- 视频处理：`OpenCV`
- 中文叠字：`Pillow`
- API：`FastAPI`
- 本地篮球检测：`Ultralytics YOLO`（免费）；远端可选 `Roboflow` / `Hugging Face`

说明：
- 当前真正的模型部分主要还是 `MediaPipe Pose`
- 动作评分和建议仍以启发式规则为主
- 篮球检测已能辅助修正离手帧，而不只是确认姿态结果
- 当前最重要的新逻辑是“最后贴手帧 -> 下一帧作为最终离手帧”

## 安装

建议使用 Python 3.10 或 3.11。当前机器上已在 Python 3.11 环境完成验证。

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install .[api]
```

如果要用本地免费篮球检测（默认 provider），安装：

```bash
pip install .[api,local]
```

如果要接远端篮球检测，再安装：

```bash
pip install .[api,remote]
```

## CLI 用法

单视频分析：

```bash
basketball-analyzer single --input shooting.mp4 --output artifacts/single
```

双视频对比：

```bash
basketball-analyzer compare --input shooting.mp4 --reference curry.mp4 --output artifacts/compare
```

本地免费篮球检测（默认模型路径 `models/basketball_yolo11.pt`，缺失时回退并自动下载 `yolov8n.pt`）：

1. 从 [Lumos-88/YOLO11-fine-tuned-for-basketball-detection](https://huggingface.co/Lumos-88/YOLO11-fine-tuned-for-basketball-detection)
   下载 `best.pt`。
2. 在项目根目录创建 `models/`，将权重保存为 `models/basketball_yolo11.pt`。

模型权重、下载缓存和第三方测试视频只保存在本机，不提交到 Git。模型页面标注 MIT，
但其基础 Ultralytics YOLO 软件另有 AGPL-3.0/企业许可要求，商业发布前需单独完成许可证评估。

```bash
basketball-analyzer single --input shooting.mp4 --output artifacts/single_local --ball-provider local
# 全身远景视频可尝试切换 COCO 通用模型
basketball-analyzer single --input testdata/gh_shot_arc.mp4 --output artifacts/single_arc --ball-provider local --local-model-path yolov8n.pt
```

接 Roboflow 远端篮球检测（可选）：

```bash
set ROBOFLOW_API_KEY=your_api_key
basketball-analyzer single --input shooting.mp4 --output artifacts/single_rf --ball-provider roboflow --roboflow-model-id basketball-game-detections/9
```

批量评估多个篮球检测模型：

```bash
# 本地模型评估（无需 API key）
basketball-analyzer ball-eval --provider local --videos shooting.mp4 shoot1.mp4 curry.mp4 --output artifacts/ball_eval_local
# 远端模型评估
set ROBOFLOW_API_KEY=your_api_key
basketball-analyzer ball-eval --provider roboflow --videos shooting.mp4 curry.mp4 --models model-a/1 model-b/1 --output artifacts/ball_eval
```

## API 用法

启动服务：

```bash
uvicorn basketball_analyzer.api:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

单视频分析：

```bash
curl -X POST http://127.0.0.1:8000/analyze/single ^
  -H "Content-Type: application/json" ^
  -d "{\"input_video\": \"D:/develop/basketball-master/shooting.mp4\", \"output_dir\": \"D:/develop/basketball-master/artifacts/single\"}"
```

## 下一步计划

短期优先级：
1. 继续扩充真实视频样本（抖音网页版截取、免费素材站），覆盖更多机位/光照
2. 针对“手部误检为球”增加候选过滤，提升微调模型在远景场景的可靠性
3. 多次投篮长视频自动切片
4. 补充批量任务、耗时日志和更多测试

中期计划：
1. 任务队列持久化与报告页
2. 把双视频对比升级成阶段化对齐与指标差异报告
3. 建立训练记录、用户历史与趋势图

长期方向：
1. 升级到服务端更强姿态模型
2. 引入自训练篮球检测模型
3. 做球星模板库与个性化训练建议
4. 发展成完整的投篮辅助 APP

## 文档

- [项目状态与路线图](D:/develop/basketball-master/docs/project-status.md)
- [工程化说明](D:/develop/basketball-master/docs/engineering-review.md)
- [远端篮球检测方案](D:/develop/basketball-master/docs/remote-ball-detection.md)
