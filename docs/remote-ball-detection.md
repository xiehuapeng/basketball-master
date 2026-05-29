# 远端篮球检测接入方案

## 当前状态

已经完成：
- 远端篮球检测提供方抽象
- `Roboflow` 与 `Hugging Face` 提供方接入
- 在离手前后少量抽帧的调用逻辑
- 单视频分析额外输出 `ball_detection_result.json`
- 多视频、多模型的批量评估 CLI 入口
- 单视频分析新增 `release_analysis`
- 输出视频已支持篮球检测框叠加
- 已支持基于“最后贴手帧 -> 下一帧离手”的最终离手帧修正

还没完成：
- 扩充更多真实视频样本，继续比较不同模型的检出率
- 将当前启发式球手分离规则升级为更稳健的时序策略

## 当前最优候选

基于目前已经跑过的真实视频：

- 第一候选：`basketball-game-detections/9`
- 第二候选：`basketball-detection-dn6fg/1`

目前结论：
- `basketball-game-detections/9` 在 `shooting.mp4` 和 `curry.mp4` 上能稳定检测到离手附近篮球
- 在 `shoot1.mp4` 上，当前几种现成 Hosted 模型都没有稳定检出
- 现成 Hosted 模型足够支撑 MVP，但还需要更多样本验证泛化能力

## 这次新增的关键能力

之前的逻辑是：
- 姿态先给出离手帧
- 如果该帧附近能看到球，就把它当最终离手帧

现在的逻辑是：
1. 先拿姿态给出一个初始离手帧
2. 如果这个姿态帧可疑，向前回溯更多原视频帧
3. 找到“球最后还贴着手”的帧
4. 把它的下一帧作为最终离手帧

这样更符合产品定义：
- `contact_frame`：最后贴手帧
- `separation_frame`：球手分离帧
- `final_release_frame`：最终离手帧

## 真实验证结果

对 `shooting.mp4` 的最新真实跑数结果：

- 原始姿态离手帧：`175`
- 最后贴手帧：`149`
- 最终离手帧：`150`
- 来源：`ball-contact-transition`

说明：
- 旧逻辑把球已经飞出去很远的帧误当成离手帧
- 新逻辑已经能把最终离手帧拉回到更接近“刚离手”的位置

参考产物：
- [analysis_result.json](D:/develop/basketball-master/artifacts/single_ball_assisted_v4/analysis_result.json)
- [single_overlay.mp4](D:/develop/basketball-master/artifacts/single_ball_assisted_v4/single_overlay.mp4)
- [149 帧截图](D:/develop/basketball-master/artifacts/single_ball_assisted_v4/overlay_frame_149.jpg)
- [150 帧截图](D:/develop/basketball-master/artifacts/single_ball_assisted_v4/overlay_frame_150.jpg)

## 当前工程如何接入

工程已经支持可插拔的远端篮球检测提供方：

- `roboflow`
- `huggingface` / `hf`

接入位置：
- [providers.py](D:/develop/basketball-master/src/basketball_analyzer/detection/providers.py:1)
- [service.py](D:/develop/basketball-master/src/basketball_analyzer/detection/service.py:1)
- [service.py](D:/develop/basketball-master/src/basketball_analyzer/service.py:1)

分析时机：
- 先跑姿态分析
- 再在离手帧附近抽样视频帧
- 必要时扩大回溯窗口
- 调用远端检测服务
- 输出 `ball_detection_result.json`
- 产出 `release_analysis`

## 当前输出字段

`ball_detection_result.json` 主要用于比较模型：
- `matched_frame_count`
- `detection_rate`
- `release_frame_detected`
- `nearest_detection_frame`
- `nearest_detection_delta`
- `best_frame`
- `best_confidence`

`release_analysis` 主要用于最终业务结果：
- `pose_release_frame`
- `contact_frame`
- `separation_frame`
- `final_release_frame`
- `source`
- `ball_candidate_wrist_distance`

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
basketball-analyzer single --input shooting.mp4 --output artifacts/single_rf --ball-provider roboflow --roboflow-model-id basketball-game-detections/9
```

批量评估：

```bash
set ROBOFLOW_API_KEY=your_api_key
basketball-analyzer ball-eval --provider roboflow --videos shooting.mp4 curry.mp4 --models basketball-game-detections/9 basketball-detection-dn6fg/1 --output artifacts/ball_eval
```

## 下一步建议

1. 用更多你自己拍的视频验证 `contact_frame -> separation_frame` 规则。
2. 继续调“贴手阈值”和“飞行阈值”，提升跨场景稳定性。
3. 后面如果现成 Hosted 模型不够，再转向自建数据集微调检测模型。
