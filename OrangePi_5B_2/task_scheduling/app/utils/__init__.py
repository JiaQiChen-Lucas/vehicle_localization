from .log import Logger
from .metrics import get_cpu_usage, calculate_cpu_usage, get_cpu_temp, get_memory_usage, get_npu_usage
from .yolov8_pose import RKNNPoolExecutor, image_pre_process, decode_result
from .response_model import ResponseModel
# from .video_stream import VideoStreamSimulator, MultiStreamManager
# from .video_stream_online import VideoStreamSimulator, MultiStreamManager
from .video_stream_ffmpeg import VideoStreamSimulator, MultiStreamManager
from .image_process import FrameInfo, stitch_images, parse_inference_result
from .detect import detect_image, choose_service_strategy_by_weight
from .byte_track import BYTETracker
from .processing import processing_video_stream
from .nacos_utils import start_nacos
from .fastapi_log import setup_logging
from .stitch_handler import stitch_handler_start
from .single_handler import single_handler_start
from .yolov8_seg import Yolov8SegExecutor, decode_seg
from .detect_yolov8m_seg import yolov8m_seg_choose_service_strategy_by_circle, yolov8m_seg_choose_service_strategy_by_weight
from .detect_yolov8s_seg import yolov8s_seg_choose_service_strategy_by_circle, yolov8s_seg_choose_service_strategy_by_weight
from .get_service_queue_length import start_background_updater_service_queue_length
