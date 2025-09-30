import numpy
import lap
import scipy


def linear_assignment(cost_matrix, thresh):
    # Linear assignment implementations with scipy and lap.lapjv
    if cost_matrix.size == 0:
        matches = numpy.empty((0, 2), dtype=int)
        unmatched_a = tuple(range(cost_matrix.shape[0]))
        unmatched_b = tuple(range(cost_matrix.shape[1]))
        return matches, unmatched_a, unmatched_b

    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = numpy.where(x < 0)[0]
    unmatched_b = numpy.where(y < 0)[0]

    return matches, unmatched_a, unmatched_b


def compute_iou(a_boxes, b_boxes):
    """
    Compute cost based on IoU
    :type a_boxes: list[tlbr] | np.ndarray
    :type b_boxes: list[tlbr] | np.ndarray
    :rtype iou | np.ndarray
    """
    iou = numpy.zeros((len(a_boxes), len(b_boxes)), dtype=numpy.float32)
    if iou.size == 0:
        return iou
    a_boxes = numpy.ascontiguousarray(a_boxes, dtype=numpy.float32)
    b_boxes = numpy.ascontiguousarray(b_boxes, dtype=numpy.float32)
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = a_boxes.T
    b2_x1, b2_y1, b2_x2, b2_y2 = b_boxes.T

    # Intersection area
    inter_area = (numpy.minimum(b1_x2[:, None], b2_x2) - numpy.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (numpy.minimum(b1_y2[:, None], b2_y2) - numpy.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (box2_area + box1_area[:, None] - inter_area + 1E-7)


def iou_distance(a_tracks, b_tracks):
    """
    Compute cost based on IoU
    :type a_tracks: list[STrack]
    :type b_tracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """

    if (len(a_tracks) > 0 and isinstance(a_tracks[0], numpy.ndarray)) \
            or (len(b_tracks) > 0 and isinstance(b_tracks[0], numpy.ndarray)):
        a_boxes = a_tracks
        b_boxes = b_tracks
    else:
        a_boxes = [track.tlbr for track in a_tracks]
        b_boxes = [track.tlbr for track in b_tracks]
    return 1 - compute_iou(a_boxes, b_boxes)  # cost matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = numpy.array([det.score for det in detections])
    det_scores = numpy.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost


class KalmanFilterXYAH:
    """
    A Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = numpy.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = numpy.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = numpy.zeros_like(mean_pos)
        mean = numpy.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        covariance = numpy.diag(numpy.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = numpy.diag(numpy.square(numpy.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        mean = numpy.dot(mean, self._motion_mat.T)
        covariance = numpy.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = numpy.diag(numpy.square(std))

        mean = numpy.dot(self._update_mat, mean)
        covariance = numpy.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrix of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [self._std_weight_position * mean[:, 3],
                   self._std_weight_position * mean[:, 3],
                   1e-2 * numpy.ones_like(mean[:, 3]),
                   self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 3],
                   self._std_weight_velocity * mean[:, 3],
                   1e-5 * numpy.ones_like(mean[:, 3]),
                   self._std_weight_velocity * mean[:, 3]]
        sqr = numpy.square(numpy.r_[std_pos, std_vel]).T

        motion_cov = [numpy.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = numpy.asarray(motion_cov)

        # print(mean)
        # print('eee')
        # print(self._motion_mat.T)
        mean = numpy.dot(mean, self._motion_mat.T)
        # print('fff')
        left = numpy.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = numpy.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             numpy.dot(covariance, self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + numpy.dot(innovation, kalman_gain.T)
        new_covariance = covariance - numpy.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        metric : str
            Distance metric.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return numpy.sum(d * d, axis=1)
        elif metric == 'maha':
            factor = numpy.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return numpy.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError('invalid distance metric')


class State:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    count = 0
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, tlwh, score, cls, keypoints):

        # wait activate
        self._tlwh = numpy.asarray(self.tlbr_to_tlwh(tlwh[:-1]), dtype=numpy.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # 关键点信息（8个关键点，每个含x,y,conf）
        self.keypoints = numpy.asarray(keypoints, dtype=numpy.float32)

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = tlwh[-1]

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != State.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) <= 0:
            return
        multi_mean = numpy.asarray([st.mean.copy() for st in tracks])
        multi_covariance = numpy.asarray([st.covariance for st in tracks])
        for i, st in enumerate(tracks):
            if st.state != State.Tracked:
                multi_mean[i][7] = 0

        multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)
        # print('eee')

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            tracks[i].mean = mean
            tracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = State.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_track.tlwh))
        self.tracklet_len = 0
        self.state = State.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        self.keypoints = new_track.keypoints  # 更新关键点

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_tlwh))
        self.state = State.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        self.keypoints = new_track.keypoints  # 更新关键点

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    def mark_lost(self):
        self.state = State.Lost

    def mark_removed(self):
        self.state = State.Removed

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        Track.count += 1
        return Track.count

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def reset_id():
        Track.count = 0

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = numpy.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = numpy.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = numpy.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BYTETracker:
    """
        BYTETracker 是一种基于检测的多目标跟踪算法，
        它通过卡尔曼滤波器预测目标位置并使用IOU进行关联，
        适用于实时视频
    """
    def __init__(self, frame_rate=30):
        # 跟踪状态管理
        self.tracked_tracks = []    # 正在跟踪的轨迹
        self.lost_tracks = []       # 丢失的轨迹
        self.removed_tracks = []    # 已移除的轨迹

        # 帧和时间管理
        self.frame_id = 0
        self.max_time_lost = int(frame_rate)    # 允许丢失的最大帧数

        # 初始化卡尔曼滤波器和ID计数器
        self.kalman_filter = KalmanFilterXYAH()
        self.reset_id()

    def update(self, boxes, object_classes, scores, keypoints):
        """
        更新跟踪器状态，处理当前帧的检测结果

        参数:
            boxes: 检测框坐标 [x1, y1, x2, y2]
            scores: 检测置信度
            object_classes: 目标类别
            keypoints: 关键点坐标 [num_detections, num_keypoints, 3]

        返回:
            numpy数组，包含跟踪结果 [x1, y1, x2, y2, track_id, score, class, detection_idx, keypoints]
        """
        self.frame_id += 1
        activated_tracks = []  # 新激活的轨迹
        re_find_tracks = []  # 重新找到的轨迹
        lost_tracks = []  # 新丢失的轨迹
        removed_tracks = []  # 新移除的轨迹

        # 为检测结果添加索引，用于后续关联
        boxes = numpy.concatenate([boxes, numpy.arange(len(boxes)).reshape(-1, 1)], axis=-1)

        # 根据置信度将检测结果分为高、中、低三部分
        indices_low = scores > 0.1
        indices_high = scores < 0.5
        indices_remain = scores > 0.5
        indices_second = numpy.logical_and(indices_low, indices_high)

        # 分层提取数据
        boxes_second = boxes[indices_second]
        boxes = boxes[indices_remain]
        scores_keep = scores[indices_remain]
        scores_second = scores[indices_second]
        cls_keep = object_classes[indices_remain]
        cls_second = object_classes[indices_second]
        keypoints_keep = keypoints[indices_remain]
        keypoints_second = keypoints[indices_second]

        # 初始化高置信度检测的轨迹
        detections = self.init_track(boxes, scores_keep, cls_keep, keypoints_keep)

        # 分离未确认的轨迹和已确认的轨迹
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)   # 未确认的轨迹（仅第一帧检测到）
            else:
                tracked_stracks.append(track)   # 已确认的轨迹

        # 第一步：使用高置信度检测进行首次关联
        track_pool = self.joint_stracks(tracked_stracks, self.lost_tracks)
        self.multi_predict(track_pool)      # 使用卡尔曼滤波器预测所有轨迹的当前位置

        # 计算轨迹与检测之间的距离矩阵并进行匈牙利匹配
        dists = self.get_dists(track_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.8)

        # 处理匹配成功的轨迹
        for tracked_i, box_i in matches:
            track = track_pool[tracked_i]
            det = detections[box_i]
            if track.state == State.Tracked:
                track.update(det, self.frame_id)    # 更新已跟踪的轨迹
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)     # 重新激活丢失的轨迹
                re_find_tracks.append(track)

        # 第二步：使用低置信度检测进行二次关联
        detections_second = self.init_track(boxes_second, scores_second, cls_second, keypoints_second)
        r_tracked_tracks = [track_pool[i] for i in u_track if track_pool[i].state == State.Tracked]

        # 计算剩余轨迹与低置信度检测之间的距离并匹配
        dists = iou_distance(r_tracked_tracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)

        # 处理二次匹配成功的轨迹
        for tracked_i, box_i in matches:
            track = r_tracked_tracks[tracked_i]
            det = detections_second[box_i]
            if track.state == State.Tracked:
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_find_tracks.append(track)

        # 标记未匹配的轨迹为丢失状态
        for it in u_track:
            track = r_tracked_tracks[it]
            if track.state != State.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        # 处理未确认的轨迹
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        # 确认匹配成功的未确认轨迹
        for tracked_i, box_i in matches:
            unconfirmed[tracked_i].update(detections[box_i], self.frame_id)
            activated_tracks.append(unconfirmed[tracked_i])

        # 移除长时间未确认的轨迹
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)

        # 第三步：初始化新的轨迹
        for new_i in u_detection:
            track = detections[new_i]
            if track.score < 0.6:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(track)

        # 第四步：更新所有轨迹的状态
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)

        # 更新轨迹列表
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == State.Tracked]
        self.tracked_tracks = self.joint_stracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = self.joint_stracks(self.tracked_tracks, re_find_tracks)
        self.lost_tracks = self.sub_stracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = self.sub_stracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_tracks)

        # 移除重复的轨迹
        self.tracked_tracks, self.lost_tracks = self.remove_duplicate_stracks(self.tracked_tracks, self.lost_tracks)

        output = []
        for track in self.tracked_tracks:
            if track.is_activated:
                # 获取检测框信息
                tlbr = track.tlbr.tolist()
                track_id = track.track_id
                score = track.score
                cls = track.cls
                idx = track.idx

                # 获取关键点并展平
                keypoints_flat = track.keypoints.flatten().tolist()  # 假设是 (K, 3) -> 展平成 K*3

                # 合并输出格式
                output.append(tlbr + [track_id, score, cls, idx] + keypoints_flat)

        return numpy.asarray(output, dtype=numpy.float32)

    @staticmethod
    def init_track(boxes, scores, cls, keypoints):
        """初始化轨迹对象，包含关键点信息"""
        return [Track(box, s, c, k) for (box, s, c, k) in zip(boxes, scores, cls, keypoints)]

    @staticmethod
    def get_dists(tracks, detections):
        """计算轨迹与检测之间的距离，融合IOU和置信度"""
        dists = iou_distance(tracks, detections)
        dists = fuse_score(dists, detections)
        return dists

    @staticmethod
    def multi_predict(tracks):
        """使用卡尔曼滤波器预测多个轨迹的下一位置"""
        Track.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """重置轨迹ID计数器"""
        Track.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """合并两个轨迹列表，避免重复ID"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """从tlista中移除tlistb中存在的轨迹"""
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """移除重复的轨迹（IOU大于阈值的轨迹对）"""
        pdist = iou_distance(stracksa, stracksb)
        pairs = numpy.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb