import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# =========================
# 1. 配置与参数优化
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE_DIR, "curry.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "pose_analysis_pro.mp4")
RELEASE_JPG = os.path.join(BASE_DIR, "release_moment.jpg")

# 提高模型复杂度（0, 1, 2），2代表最高精度，适合慢动作分析
MODEL_COMPLEXITY = 2 
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

# 自动截取参数
CLIP_PRE_SEC = 1.5    
CLIP_POST_SEC = 1.5   
SMOOTH_K = 5          # 平滑窗口

# =========================
# 2. 工具函数
# =========================
def load_chinese_font(size=24):
    for p in [r"C:\Windows\Fonts\msyh.ttc", "simhei.ttf", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def smooth_signal(data, k=5):
    if len(data) < k: return np.array(data)
    return np.convolve(data, np.ones(k)/k, mode='same')

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# =========================
# 3. 核心分析逻辑
# =========================
def analyze_shooting_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pose = mp.solutions.pose.Pose(model_complexity=MODEL_COMPLEXITY, 
                                  min_detection_confidence=MIN_DET_CONF, 
                                  min_tracking_confidence=MIN_TRK_CONF)
    
    raw_landmarks = []
    frame_idx = 0
    print("正在提取人体关键点（单遍扫描）...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            lms = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
            raw_landmarks.append((frame_idx, lms))
        frame_idx += 1
    
    cap.release()
    pose.close()
    
    if not raw_landmarks: return None

    # --- 寻找主手 ---
    # 逻辑：离手瞬间手腕高度变化剧烈的一侧
    l_wrist_y = [f[1][15][1] for f in raw_landmarks]
    r_wrist_y = [f[1][16][1] for f in raw_landmarks]
    arm = "right" if np.std(r_wrist_y) > np.std(l_wrist_y) else "left"
    S, E, W = (12, 14, 16) if arm == "right" else (11, 13, 15)
    
    # --- 锁定投篮片段 ---
    wrist_y = smooth_signal([f[1][W][1] for f in raw_landmarks])
    peak_idx_in_list = np.argmin(wrist_y) # 这里的 peak 是图像坐标最小值，即物理最高点
    
    # 截取窗口
    pre_f, post_f = int(CLIP_PRE_SEC * fps), int(CLIP_POST_SEC * fps)
    start_in_list = max(0, peak_idx_in_list - pre_f)
    end_in_list = min(len(raw_landmarks)-1, peak_idx_in_list + post_f)
    
    seg_data = raw_landmarks[start_in_list:end_in_list]
    
    # --- 离手点检测 v4 ---
    # 逻辑：在最高点前寻找伸肘速度最大的帧
    elbow_angles = [calc_angle(f[1][S][:3], f[1][E][:3], f[1][W][:3]) for f in seg_data]
    d_elbow = np.diff(smooth_signal(elbow_angles), prepend=elbow_angles[0])
    
    # 离手点通常在伸肘加速度达到峰值后的 1-3 帧（慢动作下）
    release_in_seg = np.argmax(d_elbow) - 2 
    release_in_seg = max(0, release_in_seg)
    
    return {
        "arm": arm,
        "fps": fps,
        "size": (w, h),
        "seg_data": seg_data,
        "release_idx_in_seg": release_in_seg,
        "landmarks_map": dict(raw_landmarks)
    }

# =========================
# 4. 专业视觉渲染
# =========================
def render_professional_video(analysis, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    w, h = analysis["size"]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), analysis["fps"], (w, h))
    
    arm = analysis["arm"]
    S, E, W = (12, 14, 16) if arm == "right" else (11, 13, 15)
    release_frame_global = analysis["seg_data"][analysis["release_idx_in_seg"]][0]
    
    # 预计算指标
    rel_lms = analysis["landmarks_map"][release_frame_global]
    # 计算垂直投篮角 (腕-肩 连线与水平线的夹角)
    dx = rel_lms[W][0] - rel_lms[S][0]
    dy = rel_lms[S][1] - rel_lms[W][1] # 向上为正
    release_angle = np.degrees(np.arctan2(dy, abs(dx)))
    
    elbow_angle_rel = calc_angle(rel_lms[S][:3], rel_lms[E][:3], rel_lms[W][:3])
    
    font = load_chinese_font(28)
    conn = mp.solutions.pose.POSE_CONNECTIONS

    print("正在渲染分析视频...")
    curr_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        lms = analysis["landmarks_map"].get(curr_idx)
        if lms:
            # 1. 绘制骨架 (OpenCV 绘制比 PIL 快得多)
            for pair in conn:
                p1, p2 = lms[pair[0]], lms[pair[1]]
                if p1[3] > 0.4 and p2[3] > 0.4:
                    cv2.line(frame, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), (0, 255, 0), 2)
            
            # 2. 实时标注肘部角度
            ea = calc_angle(lms[S][:3], lms[E][:3], lms[W][:3])
            cv2.putText(frame, f"{int(ea)}deg", (int(lms[E][0]*w)+10, int(lms[E][1]*h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 3. 出手时刻特殊视觉处理
            if curr_idx == release_frame_global:
                # 绘制地面垂直参考线 (从脚踝中点向上)
                ankle_x = (lms[27][0] + lms[28][0]) / 2 * w
                cv2.line(frame, (int(ankle_x), 0), (int(ankle_x), h), (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "RELEASE MOMENT", (int(w*0.4), int(h*0.9)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                cv2.imwrite(RELEASE_JPG, frame)

        # 4. 叠加中文分析面板 (仅对当前帧进行一次 PIL 转换)
        if curr_idx >= analysis["seg_data"][0][0] and curr_idx <= analysis["seg_data"][-1][0]:
            info_box = [
                f"主手: {'右' if arm=='right' else '左'}",
                f"离手肘角: {elbow_angle_rel:.1f}°",
                f"投篮弧度: {release_angle:.1f}°",
                "状态: 自动识别中" if curr_idx < release_frame_global else "状态: 随球动作分析"
            ]
            frame = draw_chinese_panel(frame, info_box, font)

        writer.write(frame)
        curr_idx += 1

    cap.release()
    writer.release()

def draw_chinese_panel(frame, lines, font):
    # 仅在左上角创建一个半透明遮罩
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for i, line in enumerate(lines):
        draw.text((25, 25 + i*35), line, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================
# 5. 主程序执行
# =========================
if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"错误: 找不到视频文件 {INPUT_VIDEO}")
    else:
        results = analyze_shooting_flow(INPUT_VIDEO)
        if results:
            render_professional_video(results, INPUT_VIDEO, OUTPUT_VIDEO)
            print(f"\n🚀 分析完成!")
            print(f"视频存储在: {OUTPUT_VIDEO}")
            print(f"出手截图在: {RELEASE_JPG}")
        else:
            print("未能识别出投篮动作，请检查视频内人物是否完整。")