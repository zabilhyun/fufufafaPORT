from functools import lru_cache
import cv2
import numpy
from tqdm import tqdm

from facefusion import inference_manager, state_manager, wording
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import DownloadScope, Fps, InferencePool, ModelOptions, ModelSet, VisionFrame
from facefusion.vision import detect_video_fps, get_video_frame, read_image

# These constants are kept for compatibility but aren't used now
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0

@lru_cache(maxsize=None)
def create_static_model_set(download_scope: DownloadScope) -> ModelSet:
    return {
        'open_nsfw': {
            'hashes': {
                'content_analyser': {
                    'url': resolve_download_url('models-3.0.0', 'open_nsfw.hash'),
                    'path': resolve_relative_path('../.assets/models/open_nsfw.hash')
                }
            },
            'sources': {
                'content_analyser': {
                    'url': resolve_download_url('models-3.0.0', 'open_nsfw.onnx'),
                    'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
                }
            },
            'size': (224, 224),
            'mean': [104, 117, 123]
        }
    }

def get_inference_pool() -> InferencePool:
    model_sources = get_model_options().get('sources')
    return inference_manager.get_inference_pool(__name__, model_sources)

def clear_inference_pool() -> None:
    inference_manager.clear_inference_pool(__name__)

def get_model_options() -> ModelOptions:
    return create_static_model_set('full').get('open_nsfw')

def pre_check() -> bool:
    model_hashes = get_model_options().get('hashes')
    model_sources = get_model_options().get('sources')
    return conditional_download_hashes(model_hashes) and conditional_download_sources(model_sources)

def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    global STREAM_COUNTER
    STREAM_COUNTER += 1
    if STREAM_COUNTER % int(video_fps) == 0:
        return analyse_frame(vision_frame)
    return False

def analyse_frame(vision_frame: VisionFrame) -> bool:
    # SKIP actual NSFW analysis — always return False
    return False

def forward(vision_frame: VisionFrame) -> float:
    # Return low score to disable NSFW filter
    return 0.0

def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    model_size = get_model_options().get('size')
    model_mean = get_model_options().get('mean')
    vision_frame = cv2.resize(vision_frame, model_size).astype(numpy.float32)
    vision_frame -= numpy.array(model_mean).astype(numpy.float32)
    vision_frame = numpy.expand_dims(vision_frame, axis=0)
    return vision_frame

@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    return False

@lru_cache(maxsize=None)
def analyse_video(video_path: str, trim_frame_start: int, trim_frame_end: int) -> bool:
    return False
