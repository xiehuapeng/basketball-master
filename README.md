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
- 接入远端篮球检测，支持 `Roboflow` 和 `Hugging Face Hosted Inference`
- 增加篮球检测批量评估入口，支持多视频、多模型对比
- 已验证当前默认优先模型为 `basketball-game-detections/9`
- 输出视频已支持篮球检测框可视化
- `release_analysis` 已支持姿态离手帧、最后贴手帧、最终离手帧、球手分离帧和分离趋势联合输出
- 已将离手判定从“看到球就确认”升级为“最后贴手帧 -> 下一帧离手”的修正规则

当前还没完成：
- 在更多真实拍摄视频上验证离手修正规则和阈值泛化能力
- 将当前启发式球手分离规则升级为更稳健的时序策略
- 多人跟踪、批量任务、前端上传页面、异步任务系统

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
- 可选远端篮球检测
  - 支持 Roboflow Hosted API
  - 支持 Hugging Face Hosted Inference
  - 支持低成本抽帧检测
  - 支持批量评估多个模型并输出汇总 JSON

## 已验证结果

当前仓库已经在本地跑通这些流程：
- `shooting.mp4` 单视频分析
- `shooting.mp4` 对比 `curry.mp4`
- FastAPI `/health`
- FastAPI `/analyze/single`
- FastAPI `/analyze/compare`
- Roboflow 远端篮球检测接入与真实视频联调

这次最新验证的关键结果：
- `shooting.mp4` 的姿态离手帧原始结果是 `175`
- 加入球辅助回溯后，最终离手帧修正为 `150`
- `149` 被识别为“最后贴手帧”
- `150` 被识别为“刚离手”的最终帧

参考产物：
- [single_ball_assisted_v4](D:/develop/basketball-master/artifacts/single_ball_assisted_v4)
- [analysis_result.json](D:/develop/basketball-master/artifacts/single_ball_assisted_v4/analysis_result.json)
- [single_overlay.mp4](D:/develop/basketball-master/artifacts/single_ball_assisted_v4/single_overlay.mp4)

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
- 远端篮球检测：`Roboflow` / `Hugging Face Hosted Inference`

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

接 Roboflow 远端篮球检测：

```bash
set ROBOFLOW_API_KEY=your_api_key
basketball-analyzer single --input shooting.mp4 --output artifacts/single_rf --ball-provider roboflow --roboflow-model-id basketball-game-detections/9
```

批量评估多个篮球检测模型：

```bash
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
1. 扩充更多真实视频样本，继续验证 `basketball-game-detections/9` 的泛化能力
2. 继续调“最后贴手帧 -> 离手帧”的阈值，让不同拍摄条件下更稳
3. 把球手分离从当前启发式规则升级成更稳健的时序策略
4. 补充批量任务、耗时日志和更多测试

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
