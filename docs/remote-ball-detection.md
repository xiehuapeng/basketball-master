# 远端篮球检测接入方案

## 当前状态

已经完成：

- 远端篮球检测提供方抽象
- `Roboflow` 与 `Hugging Face` 提供方接入
- 在离手前后少量抽帧的调用逻辑
- 单视频分析里额外输出 `ball_detection_result.json`

还没完成：

- 用真实 Hosted API key 做正式联调
- 比较不同篮球模型在真实视频上的检出率
- 把篮球检测结果正式并入离手判定主逻辑

## 适合先试的 3 个方案

### 1. Roboflow Universe 社区篮球模型

最适合 MVP 快速试跑，优先推荐。

- 适合用途：直接验证篮球是否能在离手附近稳定检出
- 推荐尝试：
  - `ball / basketball / player / hoop` 这类现成模型
  - 先挑 2 到 3 个模型做 A/B 测试
- 接入方式：Hosted API

## 2. 自己在 Roboflow 上微调篮球检测模型

当现成社区模型不稳时，这是最自然的下一步。

- 适合用途：针对手机拍摄、室内球馆、远距离小球做适配
- 建议先标注 300 到 1000 张关键帧

## 3. Hugging Face Hosted Inference 通用检测模型

更适合作为兜底或对照。

- 示例：`facebook/detr-resnet-50`
- 说明：能识别 `sports ball`，但并不是专门针对篮球视频优化

## 当前工程如何接入

工程已经支持可插拔的远端篮球检测提供方：

- `roboflow`
- `huggingface` / `hf`

接入位置：

- [providers.py](D:/develop/basketball-master/src/basketball_analyzer/detection/providers.py:1)
- [service.py](D:/develop/basketball-master/src/basketball_analyzer/detection/service.py:1)

分析时机：

- 先跑姿态分析
- 再在离手帧附近抽样少量视频帧
- 调用远端检测服务
- 输出 `ball_detection_result.json`

这样做的好处是：

- 降低远端推理成本
- 先验证“离手附近能否看到球”
- 不影响现有姿态分析主流程

## 环境变量

### Roboflow

- `ROBOFLOW_API_KEY`
- `ROBOFLOW_MODEL_ID`

### Hugging Face

- `HF_TOKEN`
- `HF_MODEL_ID`

## CLI 示例

```bash
set ROBOFLOW_API_KEY=your_api_key
basketball-analyzer single --input shooting.mp4 --output artifacts/single_rf --ball-provider roboflow --roboflow-model-id your-workspace/your-model/1
```

## 验证建议

第一轮只看 3 件事：

1. 离手前后 6 帧内有没有稳定检测到球
2. 置信度最高的球框是不是落在投篮手附近
3. 不同模型之间，哪一个在你的真实拍摄视频上漏检最少

如果这 3 件事能成立，就值得把球检测正式并入离手判定逻辑。
