#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - 坐姿信息记录模块

功能：
1. 计算关键点之间的夹角
2. 记录标准坐姿下的角度基准值
3. 实时对比当前角度与标准角度的偏差

角度定义：
- 头部前倾角：(0,6)-(6,12)，即鼻子-右肩-右髋在右肩处的夹角
- 身体倾斜角：(6,12)-(12,14)，即右肩-右髋-右膝在右髋处的夹角
- 背部弯曲程度：(6,12)线段长度，即右肩-右髋的距离，挺直时最长，弯腰时变短

使用说明：
该模块由 main.py 调用，不单独运行

作者：30740
日期：2026 年 4 月 24 日
"""

import math


class PostureRecorder:
    """坐姿信息记录器"""

    def __init__(self):
        self.baseline_head_angle = None
        self.baseline_body_angle = None
        self.baseline_back_length = None
        self.is_baseline_set = False

    @staticmethod
    def calculate_angle(keypoints, idx_a, idx_b, idx_c):
        """
        计算三个关键点在中间点处形成的夹角

        Args:
            keypoints: 关键点列表，每个元素为 {'index': int, 'x': int, 'y': int, 'confidence': float}
            idx_a: 第一个点的关键点索引
            idx_b: 顶点（中间点）的关键点索引
            idx_c: 第三个点的关键点索引

        Returns:
            angle: 夹角（度数），如果关键点缺失则返回 None
        """
        kp_dict = {kp['index']: kp for kp in keypoints}

        if idx_a not in kp_dict or idx_b not in kp_dict or idx_c not in kp_dict:
            return None

        pa = kp_dict[idx_a]
        pb = kp_dict[idx_b]
        pc = kp_dict[idx_c]

        v1_x = pa['x'] - pb['x']
        v1_y = pa['y'] - pb['y']

        v2_x = pc['x'] - pb['x']
        v2_y = pc['y'] - pb['y']

        dot = v1_x * v2_x + v1_y * v2_y

        mag1 = math.sqrt(v1_x ** 2 + v1_y ** 2)
        mag2 = math.sqrt(v2_x ** 2 + v2_y ** 2)

        if mag1 == 0 or mag2 == 0:
            return None

        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle = math.degrees(math.acos(cos_angle))

        return angle

    @staticmethod
    def calculate_segment_length(keypoints, idx_a, idx_b):
        """
        计算两个关键点之间的线段长度

        Args:
            keypoints: 关键点列表，每个元素为 {'index': int, 'x': int, 'y': int, 'confidence': float}
            idx_a: 第一个点的关键点索引
            idx_b: 第二个点的关键点索引

        Returns:
            length: 线段长度（像素），如果关键点缺失则返回 None
        """
        kp_dict = {kp['index']: kp for kp in keypoints}

        if idx_a not in kp_dict or idx_b not in kp_dict:
            return None

        pa = kp_dict[idx_a]
        pb = kp_dict[idx_b]

        dx = pa['x'] - pb['x']
        dy = pa['y'] - pb['y']

        length = math.sqrt(dx ** 2 + dy ** 2)

        return length

    def get_back_length(self, keypoints):
        """
        计算背部线段长度，即右肩(6)到右髋(12)的距离

        挺直时线段最长，弯腰时线段变短

        Args:
            keypoints: 关键点列表

        Returns:
            length: 背部线段长度（像素），缺失则返回 None
        """
        return self.calculate_segment_length(keypoints, 6, 12)

    def get_head_angle(self, keypoints):
        """
        计算头部前倾角

        顶点为右肩(6)，两条边分别为 鼻子(0)-右肩(6) 和 右肩(6)-右髋(12)

        Args:
            keypoints: 关键点列表

        Returns:
            angle: 头部前倾角（度数），缺失则返回 None
        """
        return self.calculate_angle(keypoints, 0, 6, 12)

    def get_body_angle(self, keypoints):
        """
        计算身体倾斜角

        顶点为右髋(12)，两条边分别为 右肩(6)-右髋(12) 和 右髋(12)-右膝(14)

        Args:
            keypoints: 关键点列表

        Returns:
            angle: 身体倾斜角（度数），缺失则返回 None
        """
        return self.calculate_angle(keypoints, 6, 12, 14)

    def record_baseline(self, keypoints):
        """
        记录标准坐姿的角度基准值和背部线段长度基准值

        Args:
            keypoints: 标准坐姿下的关键点列表

        Returns:
            success: 是否成功记录基准值
        """
        head_angle = self.get_head_angle(keypoints)
        body_angle = self.get_body_angle(keypoints)
        back_length = self.get_back_length(keypoints)

        if head_angle is None or body_angle is None or back_length is None:
            return False

        self.baseline_head_angle = head_angle
        self.baseline_body_angle = body_angle
        self.baseline_back_length = back_length
        self.is_baseline_set = True

        print(f"  ✅ 标准坐姿已记录")
        print(f"     头部前倾角基准值: {self.baseline_head_angle:.1f}°")
        print(f"     身体倾斜角基准值: {self.baseline_body_angle:.1f}°")
        print(f"     背部线段长度基准值: {self.baseline_back_length:.1f} px")

        return True

    def get_angle_diffs(self, keypoints):
        """
        计算当前角度与标准角度的偏差，以及背部线段长度的偏差

        Args:
            keypoints: 当前帧的关键点列表

        Returns:
            head_diff: 头部前倾角偏差（度数），正值表示前倾增大，缺失则返回 None
            body_diff: 身体倾斜角偏差（度数），正值表示倾斜增大，缺失则返回 None
            back_diff: 背部线段长度偏差（像素），正值表示背部弯曲（线段变短），缺失则返回 None
        """
        if not self.is_baseline_set:
            return None, None, None

        head_angle = self.get_head_angle(keypoints)
        body_angle = self.get_body_angle(keypoints)
        back_length = self.get_back_length(keypoints)

        head_diff = None
        body_diff = None
        back_diff = None

        if head_angle is not None:
            head_diff = head_angle - self.baseline_head_angle

        if body_angle is not None:
            body_diff = body_angle - self.baseline_body_angle

        if back_length is not None:
            back_diff = self.baseline_back_length - back_length

        return head_diff, body_diff, back_diff
