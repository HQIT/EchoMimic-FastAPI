from fastapi import FastAPI
from pydantic import BaseModel
from infer_audio2vid import run_inference, run_inference_init
from urllib.parse import urlparse

import argparse, os, yaml, tempfile, requests, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

app = FastAPI()

class InferConfig(BaseModel):
    W: int = 512
    H: int = 512
    L: int = 1200
    seed: int = 420
    facemusk_dilation_ratio: float = 0.1
    facecrop_dilation_ratio: float = 0.5
    context_frames: int = 12
    context_overlap: int = 3
    cfg: float = 2.5
    steps: int = 30
    sample_rate: int = 16000
    fps: int = 24
    device: str = "cuda"

class InferenceRequest(BaseModel):
    config: InferConfig
    ref_image_url: str
    audio_url: str

class QuotedString(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

yaml.add_representer(QuotedString, quoted_presenter)

def _create_temp_config_file(ref_image_path, audio_path):
    config_content = {
        "pretrained_base_model_path": "./pretrained_weights/sd-image-variations-diffusers/",
        "pretrained_vae_path": "./pretrained_weights/sd-vae-ft-mse/",
        "audio_model_path": "./pretrained_weights/audio_processor/whisper_tiny.pt",
        "denoising_unet_path": "./pretrained_weights/denoising_unet.pth",
        "reference_unet_path": "./pretrained_weights/reference_unet.pth",
        "face_locator_path": "./pretrained_weights/face_locator.pth",
        "motion_module_path": "./pretrained_weights/motion_module.pth",
        "inference_config": "./configs/inference/inference_v2.yaml",
        "weight_dtype": "fp16",
        "test_cases": {
            QuotedString(ref_image_path): [
                QuotedString(audio_path)
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(config_content, temp_file, default_flow_style=False)
    return temp_file.name

def _download_file(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme == 'file':
        if not os.path.exists(parsed_url.path):
            return None
        return parsed_url.path
    else:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException:
            return None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
        return temp_file.name

_ctx = {}

@app.post("/a2v")
async def infer(request: InferenceRequest):
    ref_image_path = _download_file(request.ref_image_url)
    audio_path = _download_file(request.audio_url)
    config_path = _create_temp_config_file(ref_image_path, audio_path)
    
    _logger.debug(f"ref_image_path: {ref_image_path}")
    _logger.debug(f"audio_path: {audio_path}")
    _logger.debug(f"config_path: {config_path}")

    if not ref_image_path or not audio_path:
        return {
            "ref_image_path": ref_image_path, 
            "audio_path": audio_path, 
            "output_path": None
        }

    args = argparse.Namespace(
        config=config_path,
        W=request.config.W,
        H=request.config.H,
        L=request.config.L,
        seed=request.config.seed,
        facemusk_dilation_ratio=request.config.facemusk_dilation_ratio,
        facecrop_dilation_ratio=request.config.facecrop_dilation_ratio,
        context_frames=request.config.context_frames,
        context_overlap=request.config.context_overlap,
        cfg=request.config.cfg,
        steps=request.config.steps,
        sample_rate=request.config.sample_rate,
        fps=request.config.fps,
        device=request.config.device
    )
    _logger.debug(f"args: {args}")

    output_path = run_inference(args, _ctx)
    
    return {
        "ref_image_path": ref_image_path, 
        "audio_path": audio_path, 
        "output_path": output_path
    }

@app.on_event("startup")
def _init():
    _config_path = _create_temp_config_file("", "")
    args = argparse.Namespace(config=_config_path, device="cuda")
    run_inference_init(args, _ctx)