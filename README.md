# Basketball Analyzer

一个面向“投篮辅助 APP”方向演进的篮球投篮分析项目。当前仓库已经从实验脚本整理成了可维护的工程骨架，支持单视频分析、球星动作对比、API 调用，以及可插拔的远端篮球检测接入。

## 项目状态

当前阶段：`MVP 工程版`

已经完成：

- 把原始实验脚本重构为 `src/` 下的工程化 Python 包
- 建立统一服务层、CLI 和 FastAPI API
- 跑通真实视频单次分析
- 跑通真实视频与参考球员视频对比
- 固定 `mediapipe==0.10.14`，解决当前 Windows 环境下新版本兼容问题
- 输出标准化 JSON、关键帧截图、骨架叠加视频
- 加入远端篮球检测接入层，支持 `Roboflow` 和 `Hugging Face` Hosted Inference

当前还没完成：

- 篮球检测的真实 Hosted API 联调验证
- 用“球与手分离”正式替换当前启发式离手判定
- 多人跟踪、批量任务、前端上传页面、异步任务系统

## 当前能力

- 单视频投篮分析
  - 自动提取人体姿态关键点
  - 自动判断主手
  - 自动截取投篮片段
  - 自动定位离手帧
  - 计算基础动作指标
  - 输出带骨架和文本面板的分析视频
- 双视频动作对比
  - 对齐双方离手时刻
  - 将参考球员骨架映射到用户身体尺度
  - 输出双骨架叠加对比视频
- 产品化交付入口
  - CLI
  - FastAPI 服务
  - 文档与测试骨架
- 可选远端篮球检测
  - 支持 Roboflow Hosted API
  - 支持 Hugging Face Hosted Inference
  - 当前只在离手前后少量抽帧调用，用来低成本验证篮球检测价值

## 已验证结果

当前仓库已经在本地跑通过这些流程：

- `shooting.mp4` 单视频分析
- `shooting.mp4` 对比 `curry.mp4`
- FastAPI `/health`
- FastAPI `/analyze/single`
- FastAPI `/analyze/compare`

本地验证产物示例：

- [artifacts/single_run](D:/develop/basketball-master/artifacts/single_run)
- [artifacts/api_single](D:/develop/basketball-master/artifacts/api_single)
- [artifacts/api_compare](D:/develop/basketball-master/artifacts/api_compare)
- [artifacts/api_logs](D:/develop/basketball-master/artifacts/api_logs)

## 工程框架

```text
basketball-master/
├─ src/basketball_analyzer/
│  ├─ analysis/           # 姿态启发式分析、对比映射、指标计算
│  ├─ detection/          # 远端篮球检测提供方与抽帧调用逻辑
│  ├─ rendering/          # 骨架叠加、对比视频、关键帧输出
│  ├─ api.py              # FastAPI 应用
│  ├─ cli.py              # 命令行入口
│  ├─ config.py           # 参数配置
│  ├─ models.py           # 领域模型与标准化输出对象
│  ├─ pose.py             # MediaPipe Pose 提取
│  └─ service.py          # 对外统一服务层
├─ docs/                  # 设计、状态、接入与路线图文档
├─ tests/                 # 基础单元测试
├─ overlay_pose_video.py  # 原实验脚本，保留做参考
├─ my_vs_curry_overlay.py # 原实验脚本，保留做参考
└─ basketball-Gemini.py   # 原展示脚本，保留做参考
```

原有 3 个脚本已保留，作为实验版参考；新的工程化入口在 `src/` 下。

## 当前技术选型

- 姿态估计：`MediaPipe Pose`
- 数值分析：`NumPy`
- 视频处理：`OpenCV`
- 中文叠字：`Pillow`
- API：`FastAPI`
- 远端篮球检测：`Roboflow` / `Hugging Face Hosted Inference`

说明：

- 目前真正的模型部分主要还是 `MediaPipe Pose`
- 离手检测、动作评分和建议仍以启发式规则为主
- 篮球检测已经具备接入层，但还没有在真实 Hosted API 上完成正式验证

## 安装

建议使用 Python 3.10 或 3.11。当前机器上的 Python 3.12 在 `mediapipe` 兼容性上不够稳定，项目已在 3.11 环境完成验证。

推荐：

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install .[api]
```

如果你要接远端篮球检测，再安装：

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

接 Roboflow 远端篮球检测：

```bash
set ROBOFLOW_API_KEY=your_api_key
basketball-analyzer single --input shooting.mp4 --output artifacts/single_rf --ball-provider roboflow --roboflow-model-id your-workspace/your-model/1
```

也可以直接用模块方式：

```bash
python -m basketball_analyzer.cli single --input shooting.mp4 --output artifacts/single
python -m basketball_analyzer.cli compare --input shooting.mp4 --reference curry.mp4 --output artifacts/compare
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

双视频对比：

```bash
curl -X POST http://127.0.0.1:8000/analyze/compare ^
  -H "Content-Type: application/json" ^
  -d "{\"input_video\": \"D:/develop/basketball-master/shooting.mp4\", \"reference_video\": \"D:/develop/basketball-master/curry.mp4\", \"output_dir\": \"D:/develop/basketball-master/artifacts/compare\"}"
```

Roboflow 远端篮球检测请求：

```bash
curl -X POST http://127.0.0.1:8000/analyze/single ^
  -H "Content-Type: application/json" ^
  -d "{\"input_video\": \"D:/develop/basketball-master/shooting.mp4\", \"output_dir\": \"D:/develop/basketball-master/artifacts/single_rf\", \"ball_provider\": \"roboflow\", \"roboflow_model_id\": \"your-workspace/your-model/1\"}"
```

## 下一步计划

短期优先级：

1. 用真实 Roboflow 或 Hugging Face key 跑通篮球检测
2. 评估离手前后抽帧里球的稳定检出率
3. 把“球与手分离”并入离手判定逻辑
4. 补充批量任务、耗时日志、更多测试

中期计划：

1. 接入人物跟踪，提升多人或复杂背景下稳定性
2. 把双视频对比升级成阶段化对齐与指标差异报告
3. 增加前端上传页和任务状态接口
4. 建立训练记录、用户历史与趋势图

长期方向：

1. 升级到服务端更强姿态模型
2. 引入自训练篮球检测模型
3. 做球星模板库与个性化训练建议
4. 发展成完整的投篮辅助 APP

## 文档

- [项目状态与路线图](D:/develop/basketball-master/docs/project-status.md)
- [工程化说明](D:/develop/basketball-master/docs/engineering-review.md)
- [远端篮球检测方案](D:/develop/basketball-master/docs/remote-ball-detection.md)
