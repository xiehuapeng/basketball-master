# 工程化说明与优化评估

## 当前结论

这个项目已经从“几个本地实验脚本”升级成了“可运行的 MVP 工程骨架”，并且单视频分析、球星对比分析、FastAPI 服务都已经在真实视频上跑通过。

当前最重要的判断：
- 工程化基础已经具备
- 当前主能力仍然建立在 `MediaPipe Pose + 启发式规则` 上
- 篮球检测已经开始真正影响业务结果，不再只是辅助展示
- 目前最值的投入方向仍然是 `篮球检测 + 跟踪 + 更稳健的离手修正`

## 现有功能如何实现

### 1. 单视频投篮分析

当前核心逻辑已经迁移到 `src/basketball_analyzer`：
- 用 `MediaPipe Pose` 逐帧提取 33 个关键点
- 通过左右手腕轨迹变化判断主手
- 用可见度和人体包围盒面积过滤无效帧
- 通过“腕高峰值 + 峰值显著 + 伸肘加速”截取投篮片段
- 在片段内先给出姿态离手帧
- 再结合篮球检测向前回溯，寻找“最后贴手帧 -> 下一帧离手”
- 计算动作指标并输出中文建议
- 用 `OpenCV + Pillow` 渲染骨架、篮球框和中文面板

核心模块：
- [pose.py](D:/develop/basketball-master/src/basketball_analyzer/pose.py:1)
- [analysis/core.py](D:/develop/basketball-master/src/basketball_analyzer/analysis/core.py:1)
- [service.py](D:/develop/basketball-master/src/basketball_analyzer/service.py:1)
- [rendering/single.py](D:/develop/basketball-master/src/basketball_analyzer/rendering/single.py:1)

### 2. 球星动作对比

当前对比逻辑已经被工程化封装：
- 分别分析用户视频和参考视频
- 以离手帧作为时间对齐点
- 对参考视频时间轴做插值采样
- 用髋部中心做平移，用肩宽与髋宽做缩放
- 将参考骨架映射到用户画面，输出双骨架叠加视频

核心模块：
- [analysis/comparison.py](D:/develop/basketball-master/src/basketball_analyzer/analysis/comparison.py:1)
- [rendering/comparison.py](D:/develop/basketball-master/src/basketball_analyzer/rendering/comparison.py:1)

### 3. 远端篮球检测

当前远端检测已完成接入和真实联调：
- 支持 `roboflow`
- 支持 `huggingface` / `hf`
- 支持离手附近抽帧调用
- 支持必要时扩大向前回溯窗口
- 支持批量模型评估

核心模块：
- [detection/providers.py](D:/develop/basketball-master/src/basketball_analyzer/detection/providers.py:1)
- [detection/service.py](D:/develop/basketball-master/src/basketball_analyzer/detection/service.py:1)

## 已完成的工程事项

### 工程重构

- 提炼出 `src/basketball_analyzer` 包
- 抽离分析核心、渲染模块、服务层
- 增加统一配置模型和标准输出对象
- 保留原脚本作为参考，不再作为主入口

### 产品入口

- 提供 CLI
- 提供 FastAPI 接口
- 支持标准 JSON 输出
- 支持分析视频、关键帧截图、对比视频输出

### 运行验证

- 已在本地跑通 `shooting.mp4` 单视频分析
- 已在本地跑通 `shooting.mp4` 对比 `curry.mp4`
- 已跑通 `/health`
- 已跑通 `/analyze/single`
- 已跑通 `/analyze/compare`
- 已跑通 Roboflow 真实远端篮球检测

### 最新能力升级

- 已为输出视频增加篮球检测框可视化
- 已增加 `release_analysis.contact_frame`
- 已支持用“最后贴手帧 -> 下一帧离手”修正最终离手帧
- 已补充单元测试覆盖关键修正逻辑

## 当前问题与可优化点

### 算法问题

- 当前离手修正仍然依赖启发式阈值，跨视频泛化仍需继续验证
- `shoot1.mp4` 这类视频上，现成 Hosted 模型仍然容易漏检
- 主手判断仍主要依赖腕部轨迹，遮挡场景下可能不稳
- 没有人物跟踪，多人或复杂背景下鲁棒性有限

### 产品问题

- 还没有前端上传页面
- 还没有异步任务队列和任务状态接口
- 还没有训练报告沉淀和历史趋势
- 还没有球星模板管理

### 工程问题

- 当前测试仍以基础单元测试为主，端到端集成测试还不够
- 远端推理存在 Hosted API 超时风险，需要后续补容错和重试
- 还没有完整日志、监控和错误码分层

## 下一步建议

### 第一阶段：扩大真实样本验证

1. 用更多真实拍摄视频验证离手修正逻辑。
2. 继续比较 `basketball-game-detections/9` 与备选模型的稳定性。
3. 调整贴手阈值和飞行阈值。

### 第二阶段：提升可用性

1. 增加任务状态 API 和批量分析。
2. 对比分析升级为阶段化对齐。
3. 输出更适合前端展示的报告结构。

### 第三阶段：走向投篮辅助 APP

1. 前端上传与报告展示。
2. 用户训练记录。
3. 球星模板库。
4. 个性化建议与趋势分析。
