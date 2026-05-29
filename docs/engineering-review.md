# 工程化说明与优化评估

## 当前结论

这个项目已经从“几个本地实验脚本”升级成了“可运行的 MVP 工程骨架”，并且单视频分析、球星对比分析、FastAPI 服务都已经在真实视频上跑通过。

当前最重要的判断：

- 工程化基础已经具备
- 当前主能力仍然建立在 `MediaPipe Pose + 启发式规则` 上
- 下一阶段最值得投入的是 `篮球检测 + 跟踪`，而不是先全面重写姿态分析

## 现有功能如何实现

### 1. 单视频投篮分析

当前核心逻辑已经从原始脚本迁移到 `src/basketball_analyzer`：

- 用 `MediaPipe Pose` 对视频逐帧提取 33 个关键点
- 通过左右手腕高度变化幅度判断主手
- 用可见度和人体包围盒面积过滤远处干扰人物
- 通过“手腕达到最高点 + 肘部伸展速度”锁定投篮片段
- 在片段内通过手腕高度变化和肘角变化定位离手帧
- 在离手帧附近计算动作指标并生成中文建议
- 用 `OpenCV` 绘制骨架，用 `Pillow` 绘制中文文本

核心模块：

- [pose.py](D:/develop/basketball-master/src/basketball_analyzer/pose.py:10)
- [analysis/core.py](D:/develop/basketball-master/src/basketball_analyzer/analysis/core.py:30)
- [rendering/single.py](D:/develop/basketball-master/src/basketball_analyzer/rendering/single.py:30)

### 2. 与参考球员动作对比

当前对比逻辑已经被工程化封装：

- 分别分析用户视频和参考视频
- 以离手帧作为时间对齐点
- 对参考视频时间轴做插值采样
- 用髋部中心做平移，用肩宽和髋宽均值做缩放
- 将参考骨架映射到用户画面，叠加为第二套骨架

核心模块：

- [analysis/comparison.py](D:/develop/basketball-master/src/basketball_analyzer/analysis/comparison.py:1)
- [rendering/comparison.py](D:/develop/basketball-master/src/basketball_analyzer/rendering/comparison.py:12)

### 3. 可选远端篮球检测

当前已经完成接入层，不影响主流程：

- 支持 `roboflow`
- 支持 `huggingface` / `hf`
- 先在离手帧前后抽取少量视频帧
- 调用 Hosted API
- 输出 `ball_detection_result.json`

核心模块：

- [detection/providers.py](D:/develop/basketball-master/src/basketball_analyzer/detection/providers.py:1)
- [detection/service.py](D:/develop/basketball-master/src/basketball_analyzer/detection/service.py:31)

## 已完成的事项

### 工程重构

- 提炼为 `src/basketball_analyzer` 包
- 抽离分析核心、渲染模块、服务层
- 增加统一配置模型和标准输出对象
- 保留原始脚本作为参考，不再作为主入口

### 产品入口

- 提供 CLI
- 提供 FastAPI 接口
- 支持标准 JSON 输出
- 支持分析视频、关键帧截图、对比视频输出

### 运行验证

- 已在本地跑通 `shooting.mp4` 的单视频分析
- 已在本地跑通 `shooting.mp4` 对比 `curry.mp4`
- 已跑通 `/health`
- 已跑通 `/analyze/single`
- 已跑通 `/analyze/compare`

### 依赖与兼容处理

- 固定 `mediapipe==0.10.14`
- 在 Python 3.11 环境完成验证
- 补充基础测试和编译检查

## 当前框架

### 分层结构

- `pose.py`
  - 负责姿态提取
- `analysis/`
  - 负责主手判断、片段截取、离手检测、指标计算、对比映射
- `detection/`
  - 负责远端篮球检测
- `rendering/`
  - 负责视频渲染和截图输出
- `service.py`
  - 负责对外统一调用流程
- `cli.py` / `api.py`
  - 负责交付入口

### 当前主流程

1. 读取视频
2. 提取人体姿态
3. 锁定投篮片段
4. 判断离手帧
5. 计算指标
6. 输出视频、截图和 JSON
7. 可选：离手附近抽帧调用远端篮球检测

## 目前仍存在的主要问题

### 算法问题

- 当前离手检测仍然是启发式规则，鲁棒性依赖视频拍摄角度和慢动作质量
- 主手判断只看腕部轨迹变化，容易被遮挡或特殊动作干扰
- 篮球检测还没有正式并入离手判定
- 没有多人跟踪，过滤远处人物仍然靠阈值硬筛

### 产品问题

- 还没有用户上传页面
- 还没有异步任务队列和任务状态
- 还没有用户历史记录和训练报告存档
- 还没有完整的球星模板管理

### 工程问题

- 远端篮球检测接入层已完成，但尚未做真实 API key 联调
- 当前测试主要是基础单元测试，缺少端到端集成测试
- 还没有完整日志、监控和错误码分层

## 下一步计划

### 第一阶段：验证篮球检测价值

1. 用真实 Roboflow / Hugging Face key 跑通 Hosted API
2. 比较 2 到 3 个篮球检测模型在真实投篮视频上的漏检率
3. 确认离手前后抽帧是否能稳定看到球
4. 把球检测结果并入离手判定

### 第二阶段：提升产品可用性

1. 增加任务式 API 和任务状态查询
2. 增加批量分析和日志统计
3. 对比分析升级为阶段化对齐
4. 输出更适合前端展示的报告结构

### 第三阶段：走向投篮辅助 APP

1. 前端上传与报告展示
2. 用户训练记录
3. 球星模板库
4. 个性化建议与趋势分析
