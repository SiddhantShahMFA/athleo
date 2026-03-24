from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from athleo.api.schemas import (
    ArtifactListModel,
    ExportRequestModel,
    ExportStatusModel,
    FrameMetadataModel,
    JobCreatedModel,
    JobStatusModel,
)
from athleo.jobs import JobManager, validate_export_request, validate_processing_options
from athleo.models import ExportRequest, ProcessingOptions

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


@router.post("", response_model=JobCreatedModel, status_code=202)
async def create_job(
    request: Request,
    video: UploadFile = File(...),
    model_path: str = Form("yolo26x.pt"),
    pose_model_path: str = Form("yolo26x-pose.pt"),
    conf_threshold: float = Form(0.35),
    center_alpha: float = Form(0.25),
    size_alpha: float = Form(0.15),
    lost_buffer: int = Form(10),
) -> dict:
    if not video.filename:
        raise HTTPException(status_code=400, detail="An uploaded video file is required.")
    options = ProcessingOptions(
        model_path=model_path,
        pose_model_path=pose_model_path,
        conf_threshold=conf_threshold,
        center_alpha=center_alpha,
        size_alpha=size_alpha,
        lost_buffer=lost_buffer,
    )
    try:
        validate_processing_options(options)
        manifest = get_job_manager(request).create_job(video.file, video.filename, options)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "job_id": manifest["job_id"],
        "status": manifest["status"],
        "created_at": manifest["created_at"],
        "updated_at": manifest["updated_at"],
    }


@router.get("/{job_id}", response_model=JobStatusModel)
def get_job(request: Request, job_id: str) -> dict:
    try:
        return get_job_manager(request).get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/artifacts", response_model=ArtifactListModel)
def list_artifacts(request: Request, job_id: str) -> dict:
    try:
        artifacts = get_job_manager(request).list_artifacts(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"artifacts": artifacts}


@router.get("/{job_id}/artifacts/{artifact_id}")
def download_artifact(request: Request, job_id: str, artifact_id: str) -> FileResponse:
    try:
        artifact_path = get_job_manager(request).get_artifact_path(job_id, artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=artifact_path, filename=artifact_path.name)


@router.get("/{job_id}/frames/{frame_idx}", response_model=FrameMetadataModel)
def get_frame(request: Request, job_id: str, frame_idx: int) -> dict:
    try:
        return get_job_manager(request).get_frame(job_id, frame_idx)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{job_id}/exports", response_model=ExportStatusModel, status_code=202)
def create_export(request: Request, job_id: str, payload: ExportRequestModel) -> dict:
    export_request = ExportRequest(
        mode=payload.mode,
        selected_track_id=payload.selected_track_id,
        multi_track_ids=list(payload.multi_track_ids),
        text_labels=dict(payload.text_labels),
        selected_ball_track_id=payload.selected_ball_track_id,
    )
    try:
        validate_export_request(export_request)
        return get_job_manager(request).create_export(job_id, export_request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{job_id}/exports/{export_id}", response_model=ExportStatusModel)
def get_export(request: Request, job_id: str, export_id: str) -> dict:
    try:
        return get_job_manager(request).get_export(job_id, export_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/exports/{export_id}/download")
def download_export(request: Request, job_id: str, export_id: str) -> FileResponse:
    try:
        artifact_path = get_job_manager(request).get_export_artifact_path(job_id, export_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(path=artifact_path, filename=artifact_path.name)
