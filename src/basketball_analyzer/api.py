from __future__ import annotations

from pathlib import Path

from basketball_analyzer.service import AnalysisService

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError("FastAPI dependencies are not installed. Use `pip install -e .[api]`.") from exc


class SingleAnalyzeRequest(BaseModel):
    input_video: str
    output_dir: str
    ball_provider: str = "none"
    roboflow_model_id: str = ""
    huggingface_model_id: str = ""


class CompareRequest(BaseModel):
    input_video: str
    reference_video: str
    output_dir: str


app = FastAPI(title="Basketball Analyzer API", version="0.1.0")
service = AnalysisService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze/single")
def analyze_single(request: SingleAnalyzeRequest):
    try:
        service.config.ball_detection.provider = request.ball_provider
        service.config.ball_detection.roboflow_model_id = request.roboflow_model_id
        service.config.ball_detection.huggingface_model_id = request.huggingface_model_id
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
