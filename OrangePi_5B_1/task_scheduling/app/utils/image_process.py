import numpy as np
from .yolov8_pose import letterbox, remove_letterbox
from .log import Logger
from typing import List, Tuple, Optional, Dict

class FrameInfo:
    def __init__(self, stream_id: str, frame: np.ndarray):
        self.stream_id = stream_id
        self.frame = frame

class StitchedImage:
    def __init__(self, shape: tuple = (640, 640, 3)):
        """
        初始化拼接图像对象

        参数:
            shape: 大图的尺寸，默认为 (640, 640, 3)
        """
        self.height, self.width, self.channels = shape
        self.stitched_image = np.zeros(shape, dtype=np.uint8)
        self.patch_info: List[Dict] = []  # 记录每个拼接块的信息

    def add_image(self, frameInfo: FrameInfo, row: int, col: int) -> bool:
        """
        添加一块图像到指定位置，并记录相关信息

        参数:
            frameInfo: 包含图像和 stream_id 的 FrameInfo 对象
            row: 插入的行索引（0~1）
            col: 插入的列索引（0~1）

        返回:
            成功返回 True，失败返回 False
        """

        if row < 0 or row >= 2 or col < 0 or col >= 2:
            Logger.error(f"行列越界：row={row}, col={col}")
            return False

        if frameInfo.frame.size == 0:
            Logger.error(f"帧数据为空：{frameInfo.stream_id}")
            return False

        try:
            # 缩放图像至 320x320 并保持比例
            img, padding, r = letterbox(
                frameInfo.frame,
                new_shape=(320, 320),
                color=(0, 0, 0),
                scaleup=True
            )

            # 拼接位置
            x_start = col * 320
            y_start = row * 320

            # 复制图像到大图中
            self.stitched_image[y_start:y_start + 320, x_start:x_start + 320] = img

            # 记录元信息
            self.patch_info.append({
                'stream_id': frameInfo.stream_id,
                'row': row,
                'col': col,
                'original_shape': frameInfo.frame.shape,
                'padded_shape': img.shape,
                'padding': padding,
                'scale_ratio': r
            })

            return True

        except Exception as e:
            Logger.error(f"插入图像出错：{e}")
            return False

    def get_patch_info(self) -> List[Dict]:
        """
        获取所有已拼接图像的元信息

        返回:
            包含各图像信息的列表
        """
        return self.patch_info


def stitch_images(frames: List[FrameInfo]) -> Optional[StitchedImage]:
    """
    将最多4个 320x320 图像拼接为一个 640x640 图像。

    参数:
        frames: 包含 FrameInfo 对象的列表，每个帧尺寸应为 (320, 320, 3)

    返回:
        拼接后的图像（640x640），失败时返回 None
    """
    if not frames or len(frames) == 0:
        # Logger.warning("需要至少1张图片")
        return None

    if len(frames) > 4:
        Logger.error("最多支持4张图片拼接")
        return None

    try:
        # 创建640x640的目标图像画布
        stitchedImage = StitchedImage()
        
        # 拼接逻辑：2x2布局
        for i, frame_info in enumerate(frames):
            row = i // 2  # 行索引
            col = i % 2   # 列索引

            success = stitchedImage.add_image(frame_info, row=row, col=col)
            if not success:
                Logger.error(f"拼接第 {i} 张图像失败：{frame_info.stream_id}")
                continue  # 或者 break 根据需求决定

        return stitchedImage
    except Exception as e:
        Logger.error(f"拼接图像出错: {e}")
        return None

class InferenceResult:
    def __init__(self, stream_id: str, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, keypoints: np.ndarray):
        self.stream_id = stream_id
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.keypoints = keypoints

def parse_inference_result(stitched_image: StitchedImage, result: dict) -> List[InferenceResult]:
    """
    解析拼接图像的推理结果，并将检测框和关键点映射回各原始图像坐标系

    参数:
        stitched_image (StitchedImage): 包含拼接图像及其元信息的对象
        result (dict): 推理结果，包括 'boxes', 'classes', 'scores', 'keypoints'

    返回:
        List[InferenceResult]: 每个视频流的检测结果列表
    """
    # 初始化返回结构
    parsed_result = []

    if "boxes" not in result or "classes" not in result or "scores" not in result or "keypoints" not in result:
        Logger.error("推理结果缺少必要字段")
        return parsed_result

    # 遍历所有 patch 的元信息（记录了每个小图在拼接图中的位置）
    for info in stitched_image.get_patch_info():
        stream_id = info['stream_id']
        row = info['row']
        col = info['col']
        padding = info['padding']
        scale_ratio = info['scale_ratio']

        # 计算左上角偏移量（320x320 图像在 640x640 中的位置）
        x_offset = col * 320
        y_offset = row * 320

        box_list = []
        score_list = []
        cls_list = []
        keypoints_list = []

        # 遍历所有检测目标
        for i in range(len(result["boxes"])):
            box = result["boxes"][i]
            cls = result["classes"][i]
            score = result["scores"][i]
            keypoints = result["keypoints"][i] if "keypoints" in result else None

            # 提取 box 坐标
            x1, y1, x2, y2 = box

            # 判断是否属于当前 patch（即是否落在该 patch 的区域）
            if not (x1 >= x_offset and y1 >= y_offset and x2 <= x_offset + 320 and y2 <= y_offset + 320):
                continue  # 不属于本 patch，跳过

            # 映射回原始图像坐标系
            adjusted_box = [
                (x1 - x_offset - padding[2]) / scale_ratio,  # left
                (y1 - y_offset - padding[0]) / scale_ratio,  # top
                (x2 - x_offset - padding[2]) / scale_ratio,  # right
                (y2 - y_offset - padding[0]) / scale_ratio   # bottom
            ]

            adjusted_keypoints = []
            for j in range(0, len(keypoints), 3):
                kpt_x = (keypoints[j] - x_offset - padding[2]) / scale_ratio
                kpt_y = (keypoints[j + 1] - y_offset - padding[0]) / scale_ratio
                conf = keypoints[j + 2]
                adjusted_keypoints.extend([kpt_x, kpt_y, conf])

            box_list.append(adjusted_box)
            score_list.append(score)
            cls_list.append(cls)
            keypoints_list.append(adjusted_keypoints)

        image_box = np.array(box_list, dtype=np.float32)
        image_score = np.array(score_list, dtype=np.float32)
        image_cls = np.array(cls_list, dtype=np.int32)
        image_keypoints = np.array(keypoints_list, dtype=np.float32)

        parsed_result.append(InferenceResult(stream_id=stream_id, boxes=image_box, scores=image_score, classes=image_cls, keypoints=image_keypoints))

    return parsed_result



