#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - 人体关键点绘制脚本

功能：
1. 读取 data/img/img.png 图片
2. 使用 yolo11n-pose_int8.onnx 模型进行人体关键点检测
3. 绘制人体关键点和骨架
4. 输出结果图片到 data/processed 目录

使用说明：
python3 test_draw.py

作者：30740
日期：2026 年 4 月 21 日
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

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
        self.nms_threshold = 0.45
        
        self.keypoint_name_map = {
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
        
        pad_width = (self.input_width - new_w) // 2
        pad_height = (self.input_height - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized,
            pad_height, pad_height,
            pad_width, pad_width,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        processed = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        processed = processed.astype(np.float32) / 255.0
        processed = np.transpose(processed, (2, 0, 1))
        processed = np.expand_dims(processed, axis=0)
        
        return processed, scale_factor, pad_width, pad_height
    
    def _postprocess(self, output, scale_factor, pad_width, pad_height):
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
            
        Returns:
            keypoints: 关键点列表
        """
        keypoints = []
        
        if len(output.shape) == 3:
            output = output[0]
        
        num_detections = output.shape[1]
        target_indices = [6, 12, 14]
        
        for i in range(num_detections):
            detection = output[:, i]
            
            obj_conf = detection[4]
            
            if obj_conf > self.conf_threshold:
                for kp_idx in target_indices:
                    base_idx = 5 + kp_idx * 3
                    x = detection[base_idx]
                    y = detection[base_idx + 1]
                    kp_conf = detection[base_idx + 2]
                    
                    if kp_conf > self.conf_threshold:
                        x = (x - pad_width) / scale_factor
                        y = (y - pad_height) / scale_factor
                        
                        x = max(0, min(int(x), int(self.input_width / scale_factor)))
                        y = max(0, min(int(y), int(self.input_height / scale_factor)))
                        
                        keypoints.append({
                            'index': kp_idx,
                            'x': x,
                            'y': y,
                            'confidence': kp_conf
                        })
        
        unique_keypoints = {}
        for kp in keypoints:
            if kp['index'] not in unique_keypoints or kp['confidence'] > unique_keypoints[kp['index']]['confidence']:
                unique_keypoints[kp['index']] = kp
        
        return list(unique_keypoints.values())
    
    def detect(self, image):
        """
        检测人体关键点
        
        Args:
            image: BGR 格式的 numpy 数组
            
        Returns:
            keypoints: 关键点列表
        """
        processed, scale_factor, pad_width, pad_height = self._preprocess(image)
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: processed})
        
        keypoints = self._postprocess(outputs[0], scale_factor, pad_width, pad_height)
        
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
            (6, 12), (12, 14)
        ]
        
        self.keypoint_name_map = {
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


def main():
    project_root = Path(__file__).parent.parent
    
    image_path = project_root / 'data' / 'img' / 'img1.png'
    model_path = project_root / 'model' / 'yolo11n-pose_int8.onnx'
    output_dir = project_root / 'data' / 'processed'
    output_path = output_dir / 'result1.png'
    
    print("=" * 60)
    print("树莓派 5 坐姿检测系统 - 人体关键点绘制")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在：{image_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📷 读取图像：{image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ 无法读取图像：{image_path}")
        sys.exit(1)
    
    print(f"  图像尺寸：{image.shape[1]}x{image.shape[0]}")
    
    print(f"\n🤖 加载模型：{model_path}")
    detector = PoseDetector(str(model_path))
    
    print("\n🔍 检测人体关键点...")
    keypoints = detector.detect(image)
    
    if len(keypoints) == 0:
        print("⚠️  未检测到人体关键点")
    else:
        print(f"  ✅ 检测到 {len(keypoints)} 个关键点")
        
        for kp in keypoints:
            name = detector.keypoint_name_map.get(kp['index'], f"kp{kp['index']}")
            print(f"    - {name}: ({kp['x']}, {kp['y']}), 置信度：{kp['confidence']:.3f}")
    
    print("\n🎨 绘制人体姿态...")
    visualizer = PoseVisualizer()
    result_image = visualizer.draw_pose(image, keypoints)
    
    print(f"\n💾 保存结果：{output_path}")
    cv2.imwrite(str(output_path), result_image)
    print(f"  ✅ 结果已保存到：{output_path}")
    
    print("\n" + "=" * 60)
    print("✅ 人体关键点绘制完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
