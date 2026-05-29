import os
import cv2
import numpy as np
import mediapipe as mp

from PIL import Image, ImageDraw, ImageFont

# =========================
# 配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO = os.path.join(BASE_DIR, "shooting.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "pose_overlay-shooting.mp4")
RELEASE_JPG = os.path.join(BASE_DIR, "release_frame-shooting.jpg")

SHOW_PREVIEW = False
OVERLAY_TEXT = True
HIGHLIGHT_RELEASE = True

# mediapipe pose 参数
MODEL_COMPLEXITY = 1
MIN_DET_CONF = 0.25
MIN_TRK_CONF = 0.25

# 平滑窗口
SMOOTH_K = 7

# 过滤阈值：避免远处别人
MIN_BOX_AREA = 0.06
MIN_VIS = 0.45

# 绘制阈值
DRAW_MIN_VIS = 0.30

# =========================
# ✅ 自动截取投篮片段参数（重点）
# =========================
CLIP_PRE_SEC = 2.0     # 峰值前保留多少秒
CLIP_POST_SEC = 2.5    # 峰值后保留多少秒

# “像投篮”的门槛（用于找投篮峰值事件）
MIN_RELEASE_HEIGHT = 0.06   # 手腕需明显高于肩（肩y-腕y）
PEAK_PROMINENCE = 0.012     # 峰值显著性门槛（越大越严格）
ELBOW_ACCEL_WEIGHT = 0.01   # 伸肘加速对峰值评分的加权（不要太大）

# =========================
# ✅ 离手检测 v3 参数（在截取片段内执行）
# =========================
DOWN_THR = 0.0025
DOWN_CONSEC = 3
PEAK_LEFT = 12
PEAK_RIGHT = 24
PRE_WINDOW = 18         # 在下压起点之前，向前看多少帧找“伸肘最快”
RELEASE_BACKOFF = 2     # 从“伸肘最快帧”再往前回退几帧（越大越早）

# =========================
# 中文字体（Windows）
# =========================
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

def load_chinese_font(size=26):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def put_chinese_text(frame_bgr, lines, x=20, y=20, line_h=34,
                     font=None, text_fill=(255, 255, 255),
                     stroke_fill=(0, 0, 0), stroke_width=2):
    if not lines:
        return frame_bgr
    if font is None:
        font = load_chinese_font(26)

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=text_fill,
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        yy += line_h

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# =========================
# 数学/信号处理
# =========================
def smooth_1d(x, k=7):
    x = np.array(x, dtype=float)
    if len(x) < k or k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cosine = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# =========================
# 主手判断 + 过滤
# =========================
def choose_shooting_arm(frames_landmarks):
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    ly = np.array([f[LEFT_WRIST][1] for f in frames_landmarks], dtype=float)
    ry = np.array([f[RIGHT_WRIST][1] for f in frames_landmarks], dtype=float)

    ly = smooth_1d(ly, SMOOTH_K)
    ry = smooth_1d(ry, SMOOTH_K)

    lv = np.abs(np.diff(ly, prepend=ly[0])).max()
    rv = np.abs(np.diff(ry, prepend=ry[0])).max()
    return "left" if lv > rv else "right"

def filter_frames(frames_landmarks, frame_ids, arm, min_box_area=0.06, min_vis=0.45):
    if arm == "right":
        SHOULDER, ELBOW, WRIST = 12, 14, 16
    else:
        SHOULDER, ELBOW, WRIST = 11, 13, 15

    kept_frames, kept_ids = [], []
    for f, fid in zip(frames_landmarks, frame_ids):
        xs = [p[0] for p in f]
        ys = [p[1] for p in f]
        box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

        s_vis = f[SHOULDER][3]
        e_vis = f[ELBOW][3]
        w_vis = f[WRIST][3]

        if box_area >= min_box_area and min(s_vis, e_vis, w_vis) >= min_vis:
            kept_frames.append(f)
            kept_ids.append(fid)

    return kept_frames, kept_ids

# =========================
# ✅ 自动截取投篮片段（核心新增）
# =========================
def auto_clip_shot_segment(filtered_frames, filtered_ids, arm, fps,
                           pre_sec=2.0, post_sec=2.5,
                           min_release_height=0.06, peak_prominence=0.012,
                           elbow_accel_weight=0.01):
    """
    在过滤后的序列中，自动找到“最像投篮”的峰值事件（手腕高、峰值明显、伸肘加速）
    然后按时间窗截取投篮片段，只在片段内做离手检测/指标分析。

    返回：
      seg_frames, seg_ids, seg_info(dict)
    """
    if arm == "right":
        SHOULDER, ELBOW, WRIST = 12, 14, 16
    else:
        SHOULDER, ELBOW, WRIST = 11, 13, 15

    n = len(filtered_frames)
    if n < 30:
        return filtered_frames, filtered_ids, {
            "used_clip": False,
            "reason": "有效帧过少，跳过自动截取",
            "seg_start": 0,
            "seg_end": n - 1,
            "peak_idx": int(n/2) if n else 0
        }

    wrist_y = np.array([f[WRIST][1] for f in filtered_frames], dtype=float)
    shoulder_y = np.array([f[SHOULDER][1] for f in filtered_frames], dtype=float)

    elbow_angles = []
    for f in filtered_frames:
        sh = f[SHOULDER][:3]
        el = f[ELBOW][:3]
        wr = f[WRIST][:3]
        elbow_angles.append(calc_angle(sh, el, wr))
    elbow_angles = np.array(elbow_angles, dtype=float)

    wy = smooth_1d(wrist_y, SMOOTH_K)
    sy = smooth_1d(shoulder_y, SMOOTH_K)
    ea = smooth_1d(elbow_angles, SMOOTH_K)

    # 出手高度：肩 - 腕（>0表示腕更高）
    rh = sy - wy

    # 伸肘“加速/加快”近似：肘角一阶差分
    d_ea = np.diff(ea, prepend=ea[0])

    candidates = []
    for t in range(2, n - 2):
        # 局部最高点（wy局部最小）
        if not (wy[t] <= wy[t-1] and wy[t] <= wy[t+1]):
            continue

        # 腕需明显高于肩（过滤走动摆臂）
        if rh[t] < min_release_height:
            continue

        # 峰值显著性
        left_mean = np.mean(wy[max(0, t-6):t])
        right_mean = np.mean(wy[t+1:min(n, t+7)])
        prominence = (left_mean + right_mean) / 2 - wy[t]
        if prominence < peak_prominence:
            continue

        # 峰值评分：显著性 + 峰值附近伸肘速度（越大越像投篮）
        L = max(0, t-6)
        R = min(n-1, t+6)
        elbow_speed = np.nanmax(d_ea[L:R+1])
        score = prominence + elbow_accel_weight * float(elbow_speed)
        candidates.append((score, t, prominence, elbow_speed))

    if not candidates:
        # 找不到满足门槛的峰值 -> 退化：用全局最高点
        peak_idx = int(np.argmin(wy))
        used_clip = True
        reason = "未找到强投篮峰值候选，退化为全局手腕最高点截取"
    else:
        candidates.sort(reverse=True, key=lambda x: x[0])
        peak_idx = int(candidates[0][1])
        used_clip = True
        reason = "使用投篮峰值候选（腕高+峰值显著+伸肘加速）截取"

    pre_frames = int(round(pre_sec * fps))
    post_frames = int(round(post_sec * fps))

    seg_start = max(0, peak_idx - pre_frames)
    seg_end = min(n - 1, peak_idx + post_frames)

    seg_frames = filtered_frames[seg_start:seg_end + 1]
    seg_ids = filtered_ids[seg_start:seg_end + 1]

    return seg_frames, seg_ids, {
        "used_clip": used_clip,
        "reason": reason,
        "seg_start": seg_start,
        "seg_end": seg_end,
        "peak_idx": peak_idx,
        "peak_video_frame": int(filtered_ids[peak_idx]),
        "seg_video_start": int(seg_ids[0]),
        "seg_video_end": int(seg_ids[-1]),
        "seg_len": int(len(seg_frames)),
    }

# =========================
# ✅ 离手检测 v3：先锁下压起点，再用伸肘最快定位离手
# =========================
def find_release_frame_early_v3(
    wrist_y,
    elbow_angles,
    down_thr=0.0025,
    down_consec=3,
    peak_left=12,
    peak_right=24,
    pre_window=18,
    release_backoff=2,
):
    wy = smooth_1d(wrist_y, SMOOTH_K)
    ea = smooth_1d(elbow_angles, SMOOTH_K)

    n = len(wy)
    if n < 10:
        return int(np.argmin(wy))

    peak = int(np.argmin(wy))
    L = max(1, peak - peak_left)
    R = min(n - 1, peak + peak_right)

    vy = np.diff(wy, prepend=wy[0])
    start_down = None
    for t in range(L, R - down_consec + 1):
        if all(vy[t + k] > down_thr for k in range(down_consec)):
            start_down = t
            break

    if start_down is None:
        return max(0, peak - 1)

    d_ea = np.diff(ea, prepend=ea[0])

    wl = max(0, start_down - pre_window)
    wr = start_down
    if wr <= wl + 1:
        return max(0, start_down - 2)

    best = wl + int(np.argmax(d_ea[wl:wr]))
    release = max(0, best - release_backoff)
    return int(release)

def analyze_shot_early(seg_frames, seg_ids, arm, fps):
    if arm == "right":
        SHOULDER, ELBOW, WRIST = 12, 14, 16
    else:
        SHOULDER, ELBOW, WRIST = 11, 13, 15

    elbow_angles, wrist_y = [], []
    for f in seg_frames:
        sh = f[SHOULDER][:3]
        el = f[ELBOW][:3]
        wr = f[WRIST][:3]
        elbow_angles.append(calc_angle(sh, el, wr))
        wrist_y.append(f[WRIST][1])

    if len(wrist_y) == 0:
        raise ValueError("❌ 片段内没有可用帧")

    release_idx = find_release_frame_early_v3(
        wrist_y=wrist_y,
        elbow_angles=elbow_angles,
        down_thr=DOWN_THR,
        down_consec=DOWN_CONSEC,
        peak_left=PEAK_LEFT,
        peak_right=PEAK_RIGHT,
        pre_window=PRE_WINDOW,
        release_backoff=RELEASE_BACKOFF,
    )

    release_video_frame = seg_ids[release_idx]
    return release_video_frame, release_idx, elbow_angles, wrist_y

# =========================
# 指标计算 + 建议
# =========================
def compute_metrics(frames, ids_, arm, release_idx):
    L_SH, L_EL, L_WR = 11, 13, 15
    R_SH, R_EL, R_WR = 12, 14, 16
    L_HIP, R_HIP = 23, 24

    if arm == "right":
        SHOULDER, ELBOW, WRIST = R_SH, R_EL, R_WR
        OSHOULDER, OWRIST = L_SH, L_WR
    else:
        SHOULDER, ELBOW, WRIST = L_SH, L_EL, L_WR
        OSHOULDER, OWRIST = R_SH, R_WR

    def clip(i): return max(0, min(i, len(frames) - 1))
    rel = clip(release_idx)
    f_rel = frames[rel]

    sh = f_rel[SHOULDER]
    el = f_rel[ELBOW]
    wr = f_rel[WRIST]
    osh = f_rel[OSHOULDER]
    owr = f_rel[OWRIST]
    lhip = f_rel[L_HIP]
    rhip = f_rel[R_HIP]

    elbow_angle = calc_angle(sh[:3], el[:3], wr[:3])
    release_height = (sh[1] - wr[1])

    # 随球下压幅度（简单：出手后N帧 wrist_y 最大值 - 出手时 wrist_y）
    post_n = int(max(1, round(0.35 * 30)))  # 约0.35s的概念（不强依赖fps）
    y_rel = wr[1]
    y_post = []
    for k in range(1, post_n + 1):
        y_post.append(frames[clip(rel + k)][WRIST][1])
    follow_drop = (float(np.max(y_post)) - y_rel) if len(y_post) else float("nan")

    hip_mid = [(lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2]
    sh_mid = [(sh[0] + osh[0]) / 2, (sh[1] + osh[1]) / 2]
    vx = sh_mid[0] - hip_mid[0]
    vy = hip_mid[1] - sh_mid[1]
    lateral_tilt_deg = float(np.degrees(np.arctan2(abs(vx), vy))) if vy > 1e-6 else float("nan")

    assist_wrist_sep = (owr[1] - wr[1])

    shoulder_width = abs(f_rel[R_SH][0] - f_rel[L_SH][0]) + 1e-6
    elbow_channel = abs(el[0] - sh[0]) / shoulder_width

    return {
        "release_video_frame": int(ids_[rel]),
        "elbow_angle_deg": float(elbow_angle) if not np.isnan(elbow_angle) else float("nan"),
        "release_height": float(release_height),
        "follow_through_drop": float(follow_drop) if not np.isnan(follow_drop) else float("nan"),
        "lateral_tilt_deg": float(lateral_tilt_deg) if not np.isnan(lateral_tilt_deg) else float("nan"),
        "assist_wrist_sep": float(assist_wrist_sep),
        "elbow_channel": float(elbow_channel),
        "arm": arm,
    }

def generate_coach_tips_cn(m):
    tips = []
    elbow = m["elbow_angle_deg"]
    height = m["release_height"]
    drop = m["follow_through_drop"]
    tilt = m["lateral_tilt_deg"]
    sep = m["assist_wrist_sep"]
    channel = m["elbow_channel"]

    if height >= 0.06:
        tips.append("✅ 出手点较高：继续保持手腕高于肩，出手更抗干扰。")
    elif height >= 0.04:
        tips.append("🟡 出手点中等：可把球带得更高一点，出手更稳定。")
    else:
        tips.append("⚠️ 出手点偏低：尝试提高出手点（手腕明显高于肩）。")

    if not np.isnan(drop):
        if drop >= 0.05:
            tips.append("✅ 随球下压明显：手腕有“压筐”感觉，旋转与稳定性更好。")
        elif drop >= 0.03:
            tips.append("🟡 随球一般：出手后手腕更放松下压，保持随球。")
        else:
            tips.append("⚠️ 随球不足：容易推球/平球，练习出手后手腕自然下压。")

    if not np.isnan(elbow):
        if elbow >= 150:
            tips.append("✅ 伸肘充分：发力链条更完整。")
        elif elbow >= 125:
            tips.append("🟡 伸肘尚可：出手再把“肘-腕”送出去一点。")
        else:
            tips.append("⚠️ 伸肘偏少：练“伸肘+拨腕”，力量从腿→髋→肩→肘→腕顺序传递。")

    if sep >= 0.04:
        tips.append("✅ 辅助手退得好：只扶球不发力，干扰小。")
    elif sep >= 0.015:
        tips.append("🟡 辅助手略晚：尝试更早离球，避免双手推球。")
    else:
        tips.append("⚠️ 辅助手可能干扰：辅助手与投篮手高度接近，建议更早离球。")

    if channel <= 0.45:
        tips.append("✅ 肘通道较直：方向更容易稳定。")
    elif channel <= 0.65:
        tips.append("🟡 肘通道略偏：注意“肘对筐”，减少外张。")
    else:
        tips.append("⚠️ 肘外张明显：容易左右偏，建议练直线出手通道（肘对筐）。")

    if not np.isnan(tilt):
        if tilt <= 12:
            tips.append("✅ 躯干稳定：侧倾小，出手更一致。")
        elif tilt <= 18:
            tips.append("🟡 轻微侧倾：核心收紧，出手时保持躯干更稳定。")
        else:
            tips.append("⚠️ 侧倾较大：容易偏筐，练核心稳定与直上直下起跳。")

    return tips

# =========================
# Pose 提取（第一遍）
# =========================
def extract_pose_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    frame_to_landmarks = {}
    detected_frame_ids = []
    detected_frames_landmarks = []

    idx = 0
    detected = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            detected += 1
            landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
            frame_to_landmarks[idx] = landmarks
            detected_frame_ids.append(idx)
            detected_frames_landmarks.append(landmarks)

        if SHOW_PREVIEW and idx % 5 == 0:
            cv2.imshow("extracting_pose_preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    pose.close()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print(f"✅ 第一遍完成：检测成功帧 {detected} / 总帧 {idx}")
    return frame_to_landmarks, detected_frame_ids, detected_frames_landmarks, fps, (w, h)

def save_release_frame(video_path, release_video_frame, out_jpg):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ 无法打开视频用于保存出手帧截图")
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_video_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("⚠️ 无法读取出手帧")
        return False
    cv2.imwrite(out_jpg, frame)
    return True

# =========================
# 画骨架 + 输出视频（第二遍）
# =========================
def draw_skeleton(frame, landmarks, connections, min_vis=0.3, highlight=False):
    h, w = frame.shape[:2]
    if highlight:
        line_thick, point_r = 4, 5
        line_color = (0, 0, 255)
        point_color = (0, 0, 255)
    else:
        line_thick, point_r = 2, 3
        line_color = (0, 255, 0)
        point_color = (0, 255, 0)

    for a, b in connections:
        xa, ya, _, va = landmarks[a]
        xb, yb, _, vb = landmarks[b]
        if va < min_vis or vb < min_vis:
            continue
        pa = (int(xa * w), int(ya * h))
        pb = (int(xb * w), int(yb * h))
        cv2.line(frame, pa, pb, line_color, line_thick)

    for (x, y, _, v) in landmarks:
        if v < min_vis:
            continue
        p = (int(x * w), int(y * h))
        cv2.circle(frame, p, point_r, point_color, -1)

def overlay_pose_video(video_path, out_path, frame_to_landmarks, fps, frame_size,
                       metrics=None, tips_cn=None, clip_info=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 无法打开视频：{video_path}")

    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    font = load_chinese_font(26)

    release_frame = metrics["release_video_frame"] if metrics else None

    text_lines = []
    if clip_info and clip_info.get("used_clip"):
        text_lines.append(f"自动截取片段：{clip_info['seg_video_start']}~{clip_info['seg_video_end']}（原视频帧）")
        text_lines.append(f"峰值候选帧：{clip_info['peak_video_frame']} | {clip_info['reason']}")

    if metrics is not None:
        text_lines.append(f"离手帧（片段内检测）：{metrics['release_video_frame']}")
        if not np.isnan(metrics["elbow_angle_deg"]):
            text_lines.append(f"出手肘角：{metrics['elbow_angle_deg']:.1f}°")
        text_lines.append(f"出手高度(肩-腕)：{metrics['release_height']:.3f}")
        if not np.isnan(metrics["follow_through_drop"]):
            text_lines.append(f"随球下压幅度：{metrics['follow_through_drop']:.3f}")
        if not np.isnan(metrics["lateral_tilt_deg"]):
            text_lines.append(f"躯干侧倾角：{metrics['lateral_tilt_deg']:.1f}°")
        text_lines.append(f"辅助手分离(腕高差)：{metrics['assist_wrist_sep']:.3f}")
        text_lines.append(f"肘通道(归一化)：{metrics['elbow_channel']:.3f}")

    if tips_cn:
        text_lines.append("—— 建议（前3条）——")
        for t in tips_cn[:3]:
            text_lines.append(t)

    idx = 0
    total = 0
    drawn = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total += 1
        landmarks = frame_to_landmarks.get(idx)
        highlight = (HIGHLIGHT_RELEASE and release_frame is not None and idx == release_frame)

        if landmarks is not None:
            drawn += 1
            draw_skeleton(frame, landmarks, connections, min_vis=DRAW_MIN_VIS, highlight=highlight)

            if highlight:
                cv2.putText(frame, "RELEASE", (20, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

        if OVERLAY_TEXT and text_lines:
            frame = put_chinese_text(frame, text_lines, x=20, y=20, line_h=34, font=font)

        writer.write(frame)

        if SHOW_PREVIEW and total % 2 == 0:
            cv2.imshow("pose_overlay_preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print(f"✅ 第二遍完成：输出 {out_path}")
    print(f"✅ 总帧 {total} | 绘制骨架帧 {drawn}")

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    print("🚀 输入视频：", INPUT_VIDEO)
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"❌ 找不到输入视频：{INPUT_VIDEO}")

    frame_to_landmarks, detected_ids, detected_landmarks, fps, frame_size = extract_pose_landmarks(INPUT_VIDEO)
    if len(detected_landmarks) < 10:
        raise RuntimeError("❌ 检测到的姿态帧太少，无法分析（检查光线/人物占比/遮挡）")

    arm_guess = choose_shooting_arm(detected_landmarks)
    print(f"🏀 初步判断主手：{arm_guess}")

    filtered_frames, filtered_ids = filter_frames(
        detected_landmarks, detected_ids,
        arm=arm_guess,
        min_box_area=MIN_BOX_AREA,
        min_vis=MIN_VIS
    )
    print(f"🧹 过滤后有效帧数: {len(filtered_frames)} / {len(detected_landmarks)} (MIN_BOX_AREA={MIN_BOX_AREA}, MIN_VIS={MIN_VIS})")
    if len(filtered_frames) < 30:
        print("⚠️ 过滤后帧数偏少：可将 MIN_BOX_AREA 降到 0.04~0.05 或 MIN_VIS 降到 0.35。")
        if len(filtered_frames) == 0:
            raise RuntimeError("❌ 过滤后无有效帧，无法锁定到你本人")

    # ✅ 自动截取投篮片段
    seg_frames, seg_ids, clip_info = auto_clip_shot_segment(
        filtered_frames, filtered_ids,
        arm=arm_guess,
        fps=fps,
        pre_sec=CLIP_PRE_SEC,
        post_sec=CLIP_POST_SEC,
        min_release_height=MIN_RELEASE_HEIGHT,
        peak_prominence=PEAK_PROMINENCE,
        elbow_accel_weight=ELBOW_ACCEL_WEIGHT
    )
    print(f"✂️ {clip_info.get('reason')}")
    print(f"✂️ 片段（过滤序列索引）：{clip_info.get('seg_start')}~{clip_info.get('seg_end')} | 片段长度={clip_info.get('seg_len')}")
    print(f"✂️ 片段（原视频帧号）：{clip_info.get('seg_video_start')}~{clip_info.get('seg_video_end')} | 峰值原视频帧={clip_info.get('peak_video_frame')}")

    # ✅ 只在片段内做离手检测（避免被投完走动干扰）
    release_video_frame, release_idx, elbow_angles, wrist_y = analyze_shot_early(seg_frames, seg_ids, arm_guess, fps)
    print(f"🎯 离手原视频帧号（片段内检测）: {release_video_frame}")

    metrics = compute_metrics(seg_frames, seg_ids, arm_guess, release_idx)
    tips_cn = generate_coach_tips_cn(metrics)

    print("\n📌 动作指标：")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}" if abs(v) < 100 else f"- {k}: {v}")
        else:
            print(f"- {k}: {v}")

    print("\n🧠 教练建议：")
    for t in tips_cn:
        print("-", t)

    if save_release_frame(INPUT_VIDEO, release_video_frame, RELEASE_JPG):
        print(f"\n✅ 已保存离手帧截图：{RELEASE_JPG}")

    # 输出整段原视频骨架叠加，但只高亮离手帧 & 叠字说明片段区间
    overlay_pose_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        frame_to_landmarks,
        fps=fps,
        frame_size=frame_size,
        metrics=metrics,
        tips_cn=tips_cn,
        clip_info=clip_info
    )
