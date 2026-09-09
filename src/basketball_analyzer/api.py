from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from basketball_analyzer.service import AnalysisService

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError("FastAPI dependencies are not installed. Use `pip install -e .[api]`.") from exc


class SingleAnalyzeRequest(BaseModel):
    input_video: str
    output_dir: str
    ball_provider: str = "none"
    roboflow_model_id: str | None = None
    huggingface_model_id: str | None = None
    local_model_path: str | None = None


class CompareRequest(BaseModel):
    input_video: str
    reference_video: str
    output_dir: str


@dataclass
class TaskRecord:
    task_id: str
    kind: str
    status: str = "pending"  # pending | running | done | failed
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None
    output_dir: str | None = None


app = FastAPI(title="Basketball Analyzer API", version="0.2.0")
service = AnalysisService()

_UPLOAD_ROOT = Path("artifacts/api_uploads")
_TASK_ROOT = Path("artifacts/api_tasks")
_tasks: dict[str, TaskRecord] = {}
_tasks_lock = threading.Lock()
# Analysis is CPU/GPU heavy; run one job at a time to stay predictable.
_executor = ThreadPoolExecutor(max_workers=1)


def _apply_ball_options(
    ball_provider: str,
    roboflow_model_id: str | None,
    huggingface_model_id: str | None,
    local_model_path: str | None,
) -> None:
    service.config.ball_detection.provider = ball_provider
    if roboflow_model_id:
        service.config.ball_detection.roboflow_model_id = roboflow_model_id
    if huggingface_model_id:
        service.config.ball_detection.huggingface_model_id = huggingface_model_id
    if local_model_path:
        service.config.ball_detection.local_model_path = local_model_path


def _save_upload(upload: UploadFile, task_id: str, prefix: str) -> Path:
    _UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"
    target = _UPLOAD_ROOT / f"{task_id}_{prefix}{suffix}"
    with target.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            f.write(chunk)
    return target


def _run_task(task_id: str, runner) -> None:
    with _tasks_lock:
        record = _tasks[task_id]
        record.status = "running"
    try:
        result = runner()
        with _tasks_lock:
            record.status = "done"
            record.result = result
            record.finished_at = time.time()
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        with _tasks_lock:
            record.status = "failed"
            record.error = str(exc)
            record.finished_at = time.time()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze/single")
def analyze_single(request: SingleAnalyzeRequest):
    try:
        _apply_ball_options(
            request.ball_provider,
            request.roboflow_model_id,
            request.huggingface_model_id,
            request.local_model_path,
        )
        result = service.analyze_single_video(Path(request.input_video), Path(request.output_dir))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/analyze/compare")
def analyze_compare(request: CompareRequest):
    try:
        result = service.compare_videos(
            Path(request.input_video),
            Path(request.reference_video),
            Path(request.output_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/tasks/single")
def create_single_task(
    video: UploadFile = File(...),
    ball_provider: str = Form("local"),
    local_model_path: str = Form(""),
):
    task_id = uuid.uuid4().hex[:12]
    input_path = _save_upload(video, task_id, "input")
    output_dir = _TASK_ROOT / task_id
    record = TaskRecord(task_id=task_id, kind="single", output_dir=str(output_dir))
    with _tasks_lock:
        _tasks[task_id] = record

    def runner() -> dict:
        _apply_ball_options(ball_provider, None, None, local_model_path or None)
        result = service.analyze_single_video(input_path, output_dir)
        return result.to_dict()

    _executor.submit(_run_task, task_id, runner)
    return {"task_id": task_id, "status": "pending"}


@app.post("/tasks/compare")
def create_compare_task(
    video: UploadFile = File(...),
    reference: UploadFile = File(...),
):
    task_id = uuid.uuid4().hex[:12]
    input_path = _save_upload(video, task_id, "input")
    reference_path = _save_upload(reference, task_id, "reference")
    output_dir = _TASK_ROOT / task_id
    record = TaskRecord(task_id=task_id, kind="compare", output_dir=str(output_dir))
    with _tasks_lock:
        _tasks[task_id] = record

    def runner() -> dict:
        result = service.compare_videos(input_path, reference_path, output_dir)
        return result.to_dict()

    _executor.submit(_run_task, task_id, runner)
    return {"task_id": task_id, "status": "pending"}


@app.get("/tasks")
def list_tasks():
    with _tasks_lock:
        return [
            {
                "task_id": record.task_id,
                "kind": record.kind,
                "status": record.status,
                "created_at": record.created_at,
                "finished_at": record.finished_at,
            }
            for record in sorted(_tasks.values(), key=lambda item: item.created_at, reverse=True)
        ]


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    with _tasks_lock:
        record = _tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": record.task_id,
        "kind": record.kind,
        "status": record.status,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "error": record.error,
        "result": record.result,
    }


_ARTIFACT_FILES = {
    "overlay": ("single_overlay.mp4", "video/mp4"),
    "comparison": ("comparison_overlay.mp4", "video/mp4"),
    "release": ("release_frame.jpg", "image/jpeg"),
    "result": ("analysis_result.json", "application/json"),
}


@app.get("/tasks/{task_id}/artifacts/{name}")
def get_task_artifact(task_id: str, name: str):
    with _tasks_lock:
        record = _tasks.get(task_id)
    if record is None or record.output_dir is None:
        raise HTTPException(status_code=404, detail="task not found")
    entry = _ARTIFACT_FILES.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown artifact: {name}")
    file_path = Path(record.output_dir) / entry[0]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="artifact not ready")
    return FileResponse(file_path, media_type=entry[1], filename=file_path.name)


_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>篮球投篮分析</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; max-width: 760px; margin: 24px auto; padding: 0 16px; background: #f7f8fa; color: #222; }
  h1 { font-size: 22px; }
  .card { background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  label { display: block; margin: 10px 0 4px; font-size: 14px; }
  input[type=file], select { width: 100%; }
  button { margin-top: 14px; padding: 8px 22px; border: 0; border-radius: 6px; background: #1668dc; color: #fff; cursor: pointer; }
  button:disabled { background: #9db8dc; }
  .status { margin-top: 12px; font-size: 14px; white-space: pre-wrap; }
  video, img { max-width: 100%; margin-top: 12px; border-radius: 6px; }
  a { color: #1668dc; }
</style>
</head>
<body>
<h1>🏀 篮球投篮分析</h1>
<div class="card">
  <h3>单视频分析</h3>
  <label>投篮视频（mp4）</label>
  <input id="singleFile" type="file" accept="video/*">
  <label>篮球检测</label>
  <select id="ballProvider">
    <option value="local" selected>本地免费模型（YOLO，推荐）</option>
    <option value="roboflow">Roboflow 远端模型</option>
    <option value="none">不检测篮球</option>
  </select>
  <button id="singleBtn" onclick="submitSingle()">上传并分析</button>
  <div id="singleStatus" class="status"></div>
  <div id="singleResult"></div>
</div>
<div class="card">
  <h3>球星动作对比</h3>
  <label>我的投篮视频</label>
  <input id="myFile" type="file" accept="video/*">
  <label>参考视频（球星）</label>
  <input id="refFile" type="file" accept="video/*">
  <button id="compareBtn" onclick="submitCompare()">上传并对比</button>
  <div id="compareStatus" class="status"></div>
  <div id="compareResult"></div>
</div>
<script>
async function poll(taskId, statusEl, onDone) {
  const resp = await fetch(`/tasks/${taskId}`);
  const data = await resp.json();
  if (data.status === 'done') { statusEl.textContent = '✅ 分析完成'; onDone(data); return; }
  if (data.status === 'failed') { statusEl.textContent = '❌ 分析失败：' + data.error; return; }
  statusEl.textContent = '⏳ 任务状态：' + data.status + '（分析可能需要几分钟）';
  setTimeout(() => poll(taskId, statusEl, onDone), 3000);
}
async function submitSingle() {
  const file = document.getElementById('singleFile').files[0];
  const statusEl = document.getElementById('singleStatus');
  if (!file) { statusEl.textContent = '请先选择视频文件'; return; }
  const fd = new FormData();
  fd.append('video', file);
  fd.append('ball_provider', document.getElementById('ballProvider').value);
  document.getElementById('singleBtn').disabled = true;
  statusEl.textContent = '⬆️ 上传中...';
  const resp = await fetch('/tasks/single', { method: 'POST', body: fd });
  const data = await resp.json();
  poll(data.task_id, statusEl, (task) => {
    document.getElementById('singleBtn').disabled = false;
    const tips = (task.result && task.result.tips ? task.result.tips : []).map(t => '· ' + t).join('\\n');
    document.getElementById('singleResult').innerHTML =
      `<p>教练建议：</p><pre>${tips}</pre>` +
      `<video controls src="/tasks/${task.task_id}/artifacts/overlay"></video>` +
      `<p><a href="/tasks/${task.task_id}/artifacts/result" target="_blank">查看完整 JSON 结果</a> | ` +
      `<a href="/tasks/${task.task_id}/artifacts/release" target="_blank">离手帧截图</a></p>`;
  });
  document.getElementById('singleBtn').disabled = false;
}
async function submitCompare() {
  const my = document.getElementById('myFile').files[0];
  const ref = document.getElementById('refFile').files[0];
  const statusEl = document.getElementById('compareStatus');
  if (!my || !ref) { statusEl.textContent = '请选择两个视频文件'; return; }
  const fd = new FormData();
  fd.append('video', my);
  fd.append('reference', ref);
  document.getElementById('compareBtn').disabled = true;
  statusEl.textContent = '⬆️ 上传中...';
  const resp = await fetch('/tasks/compare', { method: 'POST', body: fd });
  const data = await resp.json();
  poll(data.task_id, statusEl, (task) => {
    document.getElementById('compareBtn').disabled = false;
    document.getElementById('compareResult').innerHTML =
      `<video controls src="/tasks/${task.task_id}/artifacts/comparison"></video>`;
  });
  document.getElementById('compareBtn').disabled = false;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML
