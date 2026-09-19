#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - 实时检测模块

功能：
1. 实时调用摄像头，在屏幕上输出视频
2. 将摄像头的 1280x960 画面转换为 640x640 输入到模型中
3. 模型检测人体的三个关键点：右肩、右髋、右膝
4. 在屏幕上显示模型的标注，保持 1280x960 的显示尺寸
5. 人体所有关键点
            0: 'nose',
            1: 'right_eye',
            2: 'left_eye',
            3: 'right_ear',
            4: 'left_ear',
            5: 'right_shoulder',
            6: 'left_shoulder',
            7: 'right_elbow',
            8: 'left_elbow',
            9: 'right_wrist',
            10: 'left_wrist',
            11: 'right_hip',
            12: 'left_hip',
            13: 'right_knee',
            14: 'left_knee',
            15: 'right_ankle',
            16: 'left_ankle'

使用说明：
python3 src/posture_detection.py

作者：30740
日期：2026 年 4 月 24 日
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from picamera2 import Picamera2

try:
    import onnxruntime as ort
except ImportError as e:
    print(f"❌ 依赖库缺失：{e}")
    print("请先安装依赖：")
    print("  pip install onnxruntime numpy opencv-python")
    sys.exit(1)


class PoseDetector:
    """人体姿态检测器"""
    
    def __init__(self, model_path):
        """
        初始化姿态检测器
        
        Args:
            model_path: ONNX 模型文件路径
        """
        self.model_path = model_path
        self.session = None
        self.input_height = 640
        self.input_width = 640
        self.conf_threshold = 0.5
        
        self.keypoint_name_map = {
            0: 'nose',
            6: 'right_shoulder',
            12: 'right_hip',
            14: 'right_knee'
        }
        
        self._load_model()
    
    def _load_model(self):
        """加载 ONNX 模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        
        providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
            sess_options=session_options
        )
        
        input_info = self.session.get_inputs()[0]
        self.input_height = input_info.shape[2]
        self.input_width = input_info.shape[3]
        
        print(f"✓ 模型加载成功：{self.model_path}")
        print(f"  输入尺寸：{self.input_width}x{self.input_height}")
    
    def _preprocess(self, image):
        """
        图像预处理
        
        Args:
            image: BGR 格式的 numpy 数组
            
        Returns:
            processed_image: 预处理后的图像
            scale_factor: 缩放因子
            pad_width: 填充宽度
            pad_height: 填充高度
        """
        h, w = image.shape[:2]
        
        scale_factor = min(self.input_width / w, self.input_height / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        pad_left = (self.input_width - new_w) // 2
        pad_right = self.input_width - new_w - pad_left
        pad_top = (self.input_height - new_h) // 2
        pad_bottom = self.input_height - new_h - pad_top
        
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom,
            pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        processed = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        processed = processed.astype(np.float32) / 255.0
        processed = np.transpose(processed, (2, 0, 1))
        processed = np.expand_dims(processed, axis=0)
        
        return processed, scale_factor, pad_left, pad_top
    
    def _postprocess(self, output, scale_factor, pad_width, pad_height, original_width, original_height):
        """
        后处理模型输出
        
        YOLO 姿态模型输出格式：(1, 56, 6300)
        - 56 = 4 (bbox) + 1 (obj_conf) + 17*3 (keypoints: x, y, conf)
        - 6300 = 检测框数量
        
        Args:
            output: 模型输出
            scale_factor: 缩放因子
            pad_width: 填充宽度
            pad_height: 填充高度
            original_width: 原始图像宽度
            original_height: 原始图像高度
            
        Returns:
            keypoints: 关键点列表
        """
        if len(output.shape) == 3:
            output = output[0]

        # Accept both common layouts: [56, candidates] and [candidates, 56].
        if output.ndim == 2 and output.shape[0] != 56 and output.shape[1] == 56:
            output = output.T

        if output.ndim != 2 or output.shape[0] != 56:
            raise ValueError(f"不支持的模型输出形状: {output.shape}")

        target_indices = [0, 6, 12, 14]  # 鼻、右肩、右髋、右膝

        # 当前应用是单人坐姿检测。必须从同一个人体候选中取全部关键点，
        # 不能逐关键点跨候选取最大置信度，否则多人场景会把不同人的点拼在一起。
        valid_indices = np.flatnonzero(output[4, :] > self.conf_threshold)
        if valid_indices.size == 0:
            return []
        best_index = valid_indices[np.argmax(output[4, valid_indices])]
        detection = output[:, best_index]

        keypoints = []
        for kp_idx in target_indices:
            base_idx = 5 + kp_idx * 3
            x = detection[base_idx]
            y = detection[base_idx + 1]
            kp_conf = detection[base_idx + 2]

            if kp_conf > self.conf_threshold:
                x = (x - pad_width) / scale_factor
                y = (y - pad_height) / scale_factor

                x = max(0, min(int(x), original_width - 1))
                y = max(0, min(int(y), original_height - 1))

                keypoints.append({
                    'index': kp_idx,
                    'x': x,
                    'y': y,
                    'confidence': float(kp_conf)
                })

        return keypoints
    
    def detect(self, image):
        """
        检测人体关键点
        
        Args:
            image: BGR 格式的 numpy 数组
            
        Returns:
            keypoints: 关键点列表
        """
        h, w = image.shape[:2]
        processed, scale_factor, pad_width, pad_height = self._preprocess(image)
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: processed})
        
        keypoints = self._postprocess(outputs[0], scale_factor, pad_width, pad_height, w, h)
        
        return keypoints


class PoseVisualizer:
    """人体姿态可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.keypoint_colors = [
            (255, 0, 0), (255, 51, 0), (255, 102, 0), (255, 153, 0),
            (255, 204, 0), (255, 255, 0), (204, 255, 0), (153, 255, 0),
            (102, 255, 0), (51, 255, 0), (0, 255, 0), (0, 255, 51),
            (0, 255, 102), (0, 255, 153), (0, 255, 204), (0, 255, 255),
            (0, 204, 255), (0, 153, 255)
        ]
        
        self.skeleton = [
            (0,6),(6, 12), (12, 14)  # 鼻->右肩->右髋->右膝
        ]
        
        self.keypoint_name_map = {
            0: 'nose',
            6: 'right_shoulder',
            12: 'right_hip',
            14: 'right_knee'
        }
    
    def draw_keypoints(self, image, keypoints):
        """
        绘制关键点
        
        Args:
            image: BGR 格式的 numpy 数组
            keypoints: 关键点列表
            
        Returns:
            output_image: 绘制后的图像
        """
        output_image = image.copy()
        
        keypoint_dict = {kp['index']: kp for kp in keypoints}
        
        for kp in keypoints:
            color = self.keypoint_colors[kp['index'] % len(self.keypoint_colors)]
            
            cv2.circle(
                output_image,
                (kp['x'], kp['y']),
                6,
                color,
                -1
            )
            
            cv2.circle(
                output_image,
                (kp['x'], kp['y']),
                8,
                (255, 255, 255),
                2
            )
            
            if kp['index'] in self.keypoint_name_map:
                label = f"{self.keypoint_name_map[kp['index']]}:{kp['confidence']:.2f}"
                cv2.putText(
                    output_image,
                    label,
                    (kp['x'] + 10, kp['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        return output_image
    
    def draw_skeleton(self, image, keypoints):
        """
        绘制骨架
        
        Args:
            image: BGR 格式的 numpy 数组
            keypoints: 关键点列表
            
        Returns:
            output_image: 绘制骨架后的图像
        """
        output_image = image.copy()
        
        keypoint_dict = {kp['index']: kp for kp in keypoints}
        
        for idx1, idx2 in self.skeleton:
            if idx1 in keypoint_dict and idx2 in keypoint_dict:
                kp1 = keypoint_dict[idx1]
                kp2 = keypoint_dict[idx2]
                
                color = self.keypoint_colors[idx1 % len(self.keypoint_colors)]
                
                cv2.line(
                    output_image,
                    (kp1['x'], kp1['y']),
                    (kp2['x'], kp2['y']),
                    color,
                    2
                )
        
        return output_image
    
    def draw_pose(self, image, keypoints):
        """
        绘制完整的人体姿态（关键点 + 骨架）
        
        Args:
            image: BGR 格式的 numpy 数组
            keypoints: 关键点列表
            
        Returns:
            output_image: 绘制后的图像
        """
        output_image = self.draw_skeleton(image, keypoints)
        output_image = self.draw_keypoints(output_image, keypoints)
        
        return output_image
