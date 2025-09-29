# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml')
        validator = PoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply non-maximum suppression and return detections with high confidence scores."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]    # nkpt 代表关键点的数量
        # 如果是姿态估计任务 (is_pose == True)，sigma 设置为 OKS_SIGMA，这是一个预定义的常量，存储了 17 个关键点的标准差值（这些值通常是根据人体解剖学定义的，每个关键点有不同的标准差，表示不同的误差容忍度）。
        # 如果不是姿态估计任务，则 sigma 设为 np.ones(nkpt) / nkpt，即每个关键点的标准差都设为 1/nkpt，表示相同的误差容忍度。
        # self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.sigma = OKS_SIGMA if is_pose else np.array([0.025] * nkpt)
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[])

    def _prepare_batch(self, si, batch):
        """Prepares a batch for processing by converting keypoints to float and moving to device."""
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["kpts"] = kpts
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """Prepares and scales keypoints in a batch for pose processing."""
        predn = super()._prepare_pred(pred, pbatch)
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        # 这一行对预测的关键点坐标进行缩放（scaling），将它们从模型输入图像的尺度调整到原始图像的尺度。
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        # predn：经过准备的预测数据，包含了边界框、类别、置信度等。
        # pred_kpts：经过缩放处理的预测关键点，已经映射回原始图像坐标。
        return predn, pred_kpts

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):   # 遍历每个预测结果
            self.seen += 1      # 更新计数器，表示已经处理了多少个样本
            npr = len(pred)     # 当前图像的预测结果数量
            stat = dict(        # 初始化 stat 字典，用于存储评估过程中需要跟踪的指标
                conf=torch.zeros(0, device=self.device),    # 存储预测结果的置信度，初始为空
                pred_cls=torch.zeros(0, device=self.device),    # 存储预测的类别标签，初始为空。
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),   # 存储 True Positives（边界框的正确预测），使用布尔张量表示。
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device), # 存储 True Positives（姿态估计的正确预测），也使用布尔张量表示。
            )   # self.niou 是用于评估 IoU 的阈值数量
            pbatch = self._prepare_batch(si, batch) # 调用 _prepare_batch 方法，准备当前索引 si 对应的批次中的真实数据（ground truth），包括真实类别 cls 和真实边界框 bbox。
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)   # 真实标签的数量，即真实物体实例的数量
            stat["target_cls"] = cls
            if npr == 0:        # 处理无预测的情况
                # 如果当前图像没有任何预测结果（npr == 0），但有真实标签（nl > 0），则将空的统计数据 stat 添加到 self.stats，并更新混淆矩阵。
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:    # 如果模型只考虑单类别任务，将所有预测的类别标签设为 0。
                pred[:, 5] = 0
            #  预处理预测结果和关键点
            predn, pred_kpts = self._prepare_pred(pred, pbatch) # 调用 _prepare_pred 函数对预测结果和关键点进行处理，predn 是边界框的预测，pred_kpts 是关键点的预测
            stat["conf"] = predn[:, 4]  # 然后将预测的置信度和类别标签存储到 stat 中
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            # 评估 True Positives
            if nl:  # 如果有真实标签（nl > 0），调用 _process_batch 函数来评估预测结果是否与真实标签匹配，从而计算 True Positives。
                stat["tp"] = self._process_batch(predn, bbox, cls)  # tp 计算的是边界框的正确匹配
                stat["tp_p"] = self._process_batch(predn, bbox, cls, pred_kpts, pbatch["kpts"]) # tp_p 计算的是姿态估计（关键点）的正确匹配
                if self.args.plots:     # 如果启用了绘图选项，则更新混淆矩阵
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys(): # 将当前图像的统计数据 stat 更新到全局评估统计数据 self.stats 中
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json: # 如果设置了保存选项，将预测结果保存为 JSON 文件，以便后续分析或可视化
                self.pred_to_json(predn, batch["im_file"][si])
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.
            pred_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing predicted keypoints.
                51 corresponds to 17 keypoints each with 3 values.
            gt_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing ground truth keypoints.

        Returns:
            torch.Tensor: Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        if pred_kpts is not None and gt_kpts is not None:   # 关键点匹配
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            # ops.xyxy2xywh(gt_bboxes)：这个函数将边界框的坐标从 xyxy 格式（左上角和右下角的坐标）转换为 xywh 格式（左上角的坐标和宽度、高度）。转换后的张量是形状 [x, y, w, h]，其中 w 是宽度，h 是高度。
            # [:, 2:]：截取宽度和高度（即 w 和 h），得到一个形状为 [M, 2] 的张量，其中每一行表示边界框的宽度和高度。
            # prod(1)：对每一行计算宽度和高度的乘积，即 w * h，这就是边界框的面积。
            # * 0.53：将计算出的面积乘以系数 0.53。这个系数来源于 COCO 库的经验参数，用于关键点相似度（OKS）的计算，调整面积的尺度。
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
            '''
            # 计算每个检测框的对角线长度
            diag_len = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) ** 2 + (gt_bboxes[:, 3] - gt_bboxes[:, 1]) ** 2)
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, diag_len=diag_len)
            '''
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])
        # 调用 match_predictions：通过 match_predictions 函数，根据 IoU（或 OKS）来判断预测与真实标签是否匹配。
        # detections[:, 5] 是预测的类别标签，gt_cls 是真实的类别标签，iou 是计算出来的 IoU 或 OKS 矩阵。
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            kpts=pred_kpts,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "keypoints": p[6:],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates object detection model using COCO JSON format."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
