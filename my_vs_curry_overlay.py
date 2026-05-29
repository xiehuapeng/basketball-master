import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# =========================
# 配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MY_VIDEO = os.path.join(BASE_DIR, "shooting.mp4")           # 你的慢动作视频
CURRY_VIDEO = os.path.join(BASE_DIR, "curry.mp4")     # 库里慢动作视频
OUT_VIDEO = os.path.join(BASE_DIR, "compare_overlay.mp4")

SHOW_PREVIEW = False
MODEL_COMPLEXITY = 1
MIN_DET_CONF = 0.25
MIN_TRK_CONF = 0.25
SMOOTH_K = 7

# 过滤（防止远处别人）
MIN_BOX_AREA = 0.06
MIN_VIS = 0.45

# 自动截取投篮片段
CLIP_PRE_SEC = 2.0
CLIP_POST_SEC = 2.5
MIN_RELEASE_HEIGHT = 0.06
PEAK_PROMINENCE = 0.012
ELBOW_ACCEL_WEIGHT = 0.01

# 离手检测 v3（片段内）
DOWN_THR = 0.0025
DOWN_CONSEC = 3
PEAK_LEFT = 12
PEAK_RIGHT = 24
PRE_WINDOW = 18
RELEASE_BACKOFF = 2

# 绘制
DRAW_MIN_VIS = 0.30
DRAW_POINTS = True
DRAW_LINES = True
POINT_R = 4
LINE_THICK = 2

# 颜色（BGR）
COLOR_ME = (0, 255, 0)       # 你：绿
COLOR_CURRY = (255, 120, 0)  # 库里：蓝橙（更显眼一点），你也可以改成(255,0,0)纯蓝

# 是否镜像库里（左右手/朝向相反时用）
MIRROR_CURRY = False

# =========================
# 中文叠字（可选）
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

def put_chinese_text(frame_bgr, lines, x=20, y=20, line_h=32, font=None):
    if not lines:
        return frame_bgr
    if font is None:
        font = load_chinese_font(26)

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
        yy += line_h

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# =========================
# 工具
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
    detected_ids, detected_landmarks = [], []
    idx, detected = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            detected += 1
            landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
            frame_to_landmarks[idx] = landmarks
            detected_ids.append(idx)
            detected_landmarks.append(landmarks)

        if SHOW_PREVIEW and idx % 5 == 0:
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1

    cap.release()
    pose.close()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    return frame_to_landmarks, detected_ids, detected_landmarks, fps, (w, h), idx, detected

def auto_clip_shot_segment(filtered_frames, filtered_ids, arm, fps,
                           pre_sec=2.0, post_sec=2.5,
                           min_release_height=0.06, peak_prominence=0.012,
                           elbow_accel_weight=0.01):
    if arm == "right":
        SHOULDER, ELBOW, WRIST = 12, 14, 16
    else:
        SHOULDER, ELBOW, WRIST = 11, 13, 15

    n = len(filtered_frames)
    if n < 30:
        return filtered_frames, filtered_ids, {"used_clip": False, "seg_start": 0, "seg_end": n - 1, "peak_idx": max(0, n // 2)}

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

    rh = sy - wy  # 肩-腕

    d_ea = np.diff(ea, prepend=ea[0])

    candidates = []
    for t in range(2, n - 2):
        if not (wy[t] <= wy[t - 1] and wy[t] <= wy[t + 1]):
            continue
        if rh[t] < min_release_height:
            continue
        left_mean = np.mean(wy[max(0, t - 6):t])
        right_mean = np.mean(wy[t + 1:min(n, t + 7)])
        prominence = (left_mean + right_mean) / 2 - wy[t]
        if prominence < peak_prominence:
            continue
        L = max(0, t - 6)
        R = min(n - 1, t + 6)
        elbow_speed = float(np.nanmax(d_ea[L:R + 1]))
        score = float(prominence) + elbow_accel_weight * elbow_speed
        candidates.append((score, t))

    if candidates:
        candidates.sort(reverse=True)
        peak_idx = int(candidates[0][1])
        reason = "投篮峰值候选"
    else:
        peak_idx = int(np.argmin(wy))
        reason = "退化为手腕最高点"

    pre_frames = int(round(pre_sec * fps))
    post_frames = int(round(post_sec * fps))
    seg_start = max(0, peak_idx - pre_frames)
    seg_end = min(n - 1, peak_idx + post_frames)

    seg_frames = filtered_frames[seg_start:seg_end + 1]
    seg_ids = filtered_ids[seg_start:seg_end + 1]

    return seg_frames, seg_ids, {
        "used_clip": True,
        "reason": reason,
        "peak_idx": peak_idx,
        "seg_start": seg_start,
        "seg_end": seg_end,
        "seg_video_start": int(seg_ids[0]),
        "seg_video_end": int(seg_ids[-1]),
        "peak_video_frame": int(filtered_ids[peak_idx]),
        "seg_len": int(len(seg_frames)),
    }

def find_release_frame_early_v3(wrist_y, elbow_angles):
    wy = smooth_1d(wrist_y, SMOOTH_K)
    ea = smooth_1d(elbow_angles, SMOOTH_K)
    n = len(wy)
    if n < 10:
        return int(np.argmin(wy))

    peak = int(np.argmin(wy))
    L = max(1, peak - PEAK_LEFT)
    R = min(n - 1, peak + PEAK_RIGHT)

    vy = np.diff(wy, prepend=wy[0])
    start_down = None
    for t in range(L, R - DOWN_CONSEC + 1):
        if all(vy[t + k] > DOWN_THR for k in range(DOWN_CONSEC)):
            start_down = t
            break

    if start_down is None:
        return max(0, peak - 1)

    d_ea = np.diff(ea, prepend=ea[0])
    wl = max(0, start_down - PRE_WINDOW)
    wr = start_down
    if wr <= wl + 1:
        return max(0, start_down - 2)

    best = wl + int(np.argmax(d_ea[wl:wr]))
    release = max(0, best - RELEASE_BACKOFF)
    return int(release)

def analyze_shot_release(seg_frames, seg_ids, arm):
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

    release_idx = find_release_frame_early_v3(wrist_y, elbow_angles)
    return int(seg_ids[release_idx]), int(release_idx)

# =========================
# ✅ “库里骨架映射到你身上”
# =========================
POSE = mp.solutions.pose
CONNECTIONS = list(POSE.POSE_CONNECTIONS)

# 用肩宽+髋宽的平均作为尺度（更稳）
def body_scale(landmarks):
    L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
    def dist(a, b):
        ax, ay = landmarks[a][0], landmarks[a][1]
        bx, by = landmarks[b][0], landmarks[b][1]
        return float(np.hypot(ax - bx, ay - by))
    shoulder = dist(L_SH, R_SH)
    hip = dist(L_HIP, R_HIP)
    s = 0.5 * shoulder + 0.5 * hip
    return max(s, 1e-6)

def body_center(landmarks):
    # 用髋中点作为根节点更稳
    L_HIP, R_HIP = 23, 24
    x = 0.5 * (landmarks[L_HIP][0] + landmarks[R_HIP][0])
    y = 0.5 * (landmarks[L_HIP][1] + landmarks[R_HIP][1])
    return float(x), float(y)

def maybe_mirror_landmarks(lm):
    # 只做水平镜像：x -> 1-x
    out = [p[:] for p in lm]
    for i in range(len(out)):
        out[i][0] = 1.0 - out[i][0]
    return out

def interp_landmarks(lm_a, lm_b, alpha):
    # lm: list of [x,y,z,vis]
    out = []
    for pa, pb in zip(lm_a, lm_b):
        x = (1 - alpha) * pa[0] + alpha * pb[0]
        y = (1 - alpha) * pa[1] + alpha * pb[1]
        z = (1 - alpha) * pa[2] + alpha * pb[2]
        v = (1 - alpha) * pa[3] + alpha * pb[3]
        out.append([float(x), float(y), float(z), float(v)])
    return out

def sample_landmarks_at(frame_to_landmarks, t_float):
    # 在库里视频时间轴上做插值采样
    t0 = int(np.floor(t_float))
    t1 = int(np.ceil(t_float))
    if t0 == t1:
        return frame_to_landmarks.get(t0)
    lm0 = frame_to_landmarks.get(t0)
    lm1 = frame_to_landmarks.get(t1)
    if lm0 is None and lm1 is None:
        return None
    if lm0 is None:
        return lm1
    if lm1 is None:
        return lm0
    alpha = float(t_float - t0)
    return interp_landmarks(lm0, lm1, alpha)

def map_curry_to_me(lm_me, lm_curry):
    """
    将库里骨架从“库里坐标系”映射到“你坐标系”
    1) 以髋中点为中心做平移
    2) 用尺度（肩+髋）做缩放
    """
    if lm_me is None or lm_curry is None:
        return None

    s_me = body_scale(lm_me)
    s_cu = body_scale(lm_curry)
    scale = s_me / s_cu

    cx_me, cy_me = body_center(lm_me)
    cx_cu, cy_cu = body_center(lm_curry)

    out = []
    for x, y, z, v in lm_curry:
        x2 = (x - cx_cu) * scale + cx_me
        y2 = (y - cy_cu) * scale + cy_me
        out.append([float(x2), float(y2), float(z), float(v)])
    return out

def draw_skeleton(frame, landmarks, color=(0,255,0), min_vis=0.3):
    h, w = frame.shape[:2]

    if DRAW_LINES:
        for a, b in CONNECTIONS:
            xa, ya, _, va = landmarks[a]
            xb, yb, _, vb = landmarks[b]
            if va < min_vis or vb < min_vis:
                continue
            pa = (int(xa * w), int(ya * h))
            pb = (int(xb * w), int(yb * h))
            cv2.line(frame, pa, pb, color, LINE_THICK)

    if DRAW_POINTS:
        for (x, y, _, v) in landmarks:
            if v < min_vis:
                continue
            p = (int(x * w), int(y * h))
            cv2.circle(frame, p, POINT_R, color, -1)

# =========================
# 主流程
# =========================
def process_video(video_path, tag=""):
    f2lm, det_ids, det_lms, fps, size, total_frames, detected = extract_pose_landmarks(video_path)
    if len(det_lms) < 15:
        raise RuntimeError(f"❌ {tag} 检测到的姿态帧太少：{len(det_lms)}")
    arm = choose_shooting_arm(det_lms)
    flt_frames, flt_ids = filter_frames(det_lms, det_ids, arm, MIN_BOX_AREA, MIN_VIS)

    seg_frames, seg_ids, clip_info = auto_clip_shot_segment(
        flt_frames, flt_ids, arm, fps,
        pre_sec=CLIP_PRE_SEC, post_sec=CLIP_POST_SEC,
        min_release_height=MIN_RELEASE_HEIGHT,
        peak_prominence=PEAK_PROMINENCE,
        elbow_accel_weight=ELBOW_ACCEL_WEIGHT
    )
    rel_frame, rel_idx = analyze_shot_release(seg_frames, seg_ids, arm)

    return {
        "frame_to_landmarks": f2lm,
        "fps": fps,
        "size": size,
        "arm": arm,
        "clip_info": clip_info,
        "release_frame": rel_frame,
        "total_frames": total_frames,
    }

def main():
    if not os.path.exists(MY_VIDEO):
        raise FileNotFoundError(f"找不到 {MY_VIDEO}")
    if not os.path.exists(CURRY_VIDEO):
        raise FileNotFoundError(f"找不到 {CURRY_VIDEO}")

    print("🎬 解析你的视频...")
    me = process_video(MY_VIDEO, tag="ME")
    print(f"✅ 你：fps={me['fps']:.1f} size={me['size']} arm={me['arm']} release={me['release_frame']} clip={me['clip_info'].get('seg_video_start')}~{me['clip_info'].get('seg_video_end')}")

    print("🎬 解析库里视频...")
    cu = process_video(CURRY_VIDEO, tag="CURRY")
    print(f"✅ 库里：fps={cu['fps']:.1f} size={cu['size']} arm={cu['arm']} release={cu['release_frame']} clip={cu['clip_info'].get('seg_video_start')}~{cu['clip_info'].get('seg_video_end')}")

    # 输出以“你的分辨率”为准
    w, h = me["size"]
    out_fps = me["fps"] if me["fps"] > 0 else 30.0

    cap = cv2.VideoCapture(MY_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError("无法打开你的原视频")

    writer = cv2.VideoWriter(
        OUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (w, h)
    )

    font = load_chinese_font(26)

    # ✅ 时间对齐：以离手帧对齐
    # 对于你的每一帧 idx，映射到库里时间：t_curry = curry_release + (idx - my_release) * (curry_fps / my_fps)
    ratio = (cu["fps"] / me["fps"]) if (me["fps"] > 0 and cu["fps"] > 0) else 1.0

    my_release = me["release_frame"]
    curry_release = cu["release_frame"]

    idx = 0
    drawn_me, drawn_cu = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm_me = me["frame_to_landmarks"].get(idx)
        if lm_me is not None:
            draw_skeleton(frame, lm_me, COLOR_ME, min_vis=DRAW_MIN_VIS)
            drawn_me += 1

        # 库里在你的这帧应该对应的时间点（可插值）
        t_curry = curry_release + (idx - my_release) * ratio
        lm_cu = sample_landmarks_at(cu["frame_to_landmarks"], t_curry)
        if lm_cu is not None:
            if MIRROR_CURRY:
                lm_cu = maybe_mirror_landmarks(lm_cu)
            # 映射到你身上
            if lm_me is not None:
                lm_cu_mapped = map_curry_to_me(lm_me, lm_cu)
            else:
                lm_cu_mapped = lm_cu  # 没有你的pose就先不映射（也可以选择跳过）
            if lm_cu_mapped is not None:
                draw_skeleton(frame, lm_cu_mapped, COLOR_CURRY, min_vis=DRAW_MIN_VIS)
                drawn_cu += 1

        # 中文叠字
        lines = [
            f"你(绿) vs 库里(蓝) — 离手对齐",
            f"你的离手帧: {my_release} | 库里离手帧: {curry_release}",
        ]
        frame = put_chinese_text(frame, lines, x=20, y=20, line_h=32, font=font)

        writer.write(frame)

        if SHOW_PREVIEW and idx % 2 == 0:
            cv2.imshow("compare_preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print(f"✅ 输出完成：{OUT_VIDEO}")
    print(f"✅ 你的骨架绘制帧：{drawn_me} | 库里骨架叠加帧：{drawn_cu}")

if __name__ == "__main__":
    main()
