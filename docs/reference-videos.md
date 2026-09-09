# Reference Videos

## Local test videos (testdata/)

新增于 2026-07，来源为 GitHub CV 项目样例，仅保存在本机用于算法评估，`testdata/`
已加入 `.gitignore`，不会随仓库分发：

- `testdata/gh_shot_arc.mp4`
  - 来源：`mathmerizing/Basketball_Shot_Analysis`（GitHub, `test.mp4`）
  - 许可：源仓库声明 Apache-2.0；再次分发时仍需保留相应许可与归属信息
  - 内容：后院单次左手投篮，1080x1920 竖屏，51 帧
  - 用途：全身远景单次投篮；已暴露篮球微调模型"把手误检为球"的已知限制，该视频建议 `--local-model-path yolov8n.pt`
- `testdata/gh_outdoor_shots.mp4`
  - 来源：`avishah3/AI-Basketball-Shot-Detection-Tracker`（GitHub, `video_test_5.mp4`）
  - 许可：源仓库未声明许可证，因此仅作本地验证，不提交、不再分发
  - 内容：室内球馆连续 20+ 次跳投合集，960x540 横屏，3378 帧
  - 用途：长视频多次投篮场景（当前流程按单次投篮假设处理，需要先切片）
- `testdata/gh_outdoor_clip1/2/3.mp4`
  - 由 `gh_outdoor_shots.mp4` 切出的单次投篮片段（帧 110-260 / 590-740 / 1040-1190）
  - 用途：远景小人物回归用例；clip3 覆盖"球贴手贯穿 pose 帧需向后推进"场景

## Curry side-view shooting reference

- Source: Douyin
- Creator: @Accepted
- Title: 深度解析斯蒂芬库里在8月2日训练营中的完整投篮训练！
- Link: https://www.douyin.com/video/7267496565156203836
- Search-modal link: https://www.douyin.com/search/%E5%BA%93%E9%87%8C%20%E8%AE%AD%E7%BB%83%20%E4%B8%89%E5%88%86%20%E4%BE%A7%E9%9D%A2%20%E5%85%A8%E8%BA%AB%20%E5%87%BA%E6%89%8B%E7%82%B9%20%E8%BF%9C%E6%99%AF?modal_id=7267496565156203836&type=general
- Duration: 11:29
- Fit for this project: Better primary reference than the close-up clip. It shows real Stephen Curry in training from a farther side/front-side angle, with the full shooting chain and ball release point visible.
- Caveat: Some playback UI can cover the very bottom of the frame in the browser, but the source video itself is a better full-body reference than the closer side-view tutorial clip.
- Suggested use: Use this as the main external reference link for full-body release-point review.

## Curry close side-view shooting reference

- Source: Douyin
- Creator: @铁林投篮
- Title: 库里投篮视频教程里各距离的正面、侧面慢动作，练习照着对比。
- Link: https://www.douyin.com/video/7579465374937156916
- Search-modal link: https://www.douyin.com/search/%E5%BA%93%E9%87%8C%20%E4%BE%A7%E8%BA%AB%20%E6%8A%95%E7%AF%AE%20%E6%85%A2%E5%8A%A8%E4%BD%9C%20%E6%97%A0%E9%81%AE%E6%8C%A1?modal_id=7579465374937156916&type=general
- Duration: 00:58
- Fit for this project: Good side-view reference. The visible side-view segment has Curry's full body and ball path unobstructed, with a clean training-gym background.
- Suggested use: Keep as an external reference link unless you have permission to store the actual video file in the repository.

## Curry left-side shooting reference

- Source: Douyin
- Creator: @Eric许涛（1%）
- Title: 看完视频！以后防库里就给左侧45度！
- Link: https://www.douyin.com/video/7540255870489627961
- Search-modal link: https://www.douyin.com/search/%E5%BA%93%E9%87%8C%20%E5%B7%A6%E5%BA%95%E8%A7%92%20%E4%B8%89%E5%88%86%20%E8%AE%AD%E7%BB%83%20%E5%85%A8%E8%BA%AB%20%E5%87%BA%E6%89%8B?modal_id=7540255870489627961&type=general
- Duration: 02:49
- Fit for this project: Good left-side / left-45-degree reference. The visible shooting sequence uses a farther event-court camera angle, with Curry's full body and the ball release path visible.
- Caveat: There is on-screen comment text near the top, but it does not block Curry's body or release point.
- Suggested use: Use as a secondary external reference when comparing left-side shooting mechanics.
