#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - 距离检测模块

功能：
1. 通过手动标注屏幕参考点(hand)和20cm参考线段来建立像素与实际距离的比例
2. 实时计算鼻子(nose)到屏幕参考点(hand)的实际距离
3. 在画面上绘制距离信息和颜色提示

使用说明：
该模块由 main.py 调用，不单独运行

作者：30740
日期：2026 年 4 月 26 日
"""

import math
import cv2


REFERENCE_LENGTH_CM = 20.0  # 参考线段实际长度（厘米）


class DistanceDetector:
    """头部到屏幕距离检测器"""

    # 标定状态
    CALIB_NONE = 0       # 未开始标定
    CALIB_HAND = 1       # 等待标注屏幕参考点(hand)
    CALIB_REF_START = 2  # 等待标注20cm参考线段起点
    CALIB_REF_END = 3    # 等待标注20cm参考线段终点
    CALIB_DONE = 4       # 标定完成

    def __init__(self):
        self.calib_state = self.CALIB_NONE
        self.hand_point = None          # 屏幕参考点 (x, y)
        self.ref_start = None           # 20cm参考线段起点 (x, y)
        self.ref_end = None             # 20cm参考线段终点 (x, y)
        self.pixels_per_cm = None       # 像素/厘米 比例
        self.is_calibrated = False

    def start_calibration(self):
        """开始标定流程"""
        self.calib_state = self.CALIB_HAND
        self.hand_point = None
        self.ref_start = None
        self.ref_end = None
        self.pixels_per_cm = None
        self.is_calibrated = False
        print("📏 距离标定开始")
        print("  步骤 1/3：请点击屏幕上电脑屏幕的参考位置")

    def reset(self):
        """重置标定状态"""
        self.calib_state = self.CALIB_NONE
        self.hand_point = None
        self.ref_start = None
        self.ref_end = None
        self.pixels_per_cm = None
        self.is_calibrated = False

    def handle_click(self, x, y):
        """
        处理鼠标点击事件，推进标定流程

        Args:
            x: 点击的 x 坐标
            y: 点击的 y 坐标

        Returns:
            calib_done: 是否标定完成
        """
        if self.calib_state == self.CALIB_HAND:
            self.hand_point = (x, y)
            self.calib_state = self.CALIB_REF_START
            print(f"  屏幕参考点已标注: ({x}, {y})")
            print("  步骤 2/3：请点击20cm参考线段的起点")
            return False

        elif self.calib_state == self.CALIB_REF_START:
            self.ref_start = (x, y)
            self.calib_state = self.CALIB_REF_END
            print(f"  参考线段起点已标注: ({x}, {y})")
            print("  步骤 3/3：请点击20cm参考线段的终点")
            return False

        elif self.calib_state == self.CALIB_REF_END:
            self.ref_end = (x, y)
            ref_px = self._pixel_distance(self.ref_start, self.ref_end)
            if ref_px < 5:
                print("  参考线段太短，请重新标注")
                self.ref_start = None
                self.ref_end = None
                self.calib_state = self.CALIB_REF_START
                print("  步骤 2/3：请点击20cm参考线段的起点")
                return False
            self.pixels_per_cm = ref_px / REFERENCE_LENGTH_CM
            self.is_calibrated = True
            self.calib_state = self.CALIB_DONE
            print(f"  参考线段终点已标注: ({x}, {y})")
            print(f"  20cm 参考线段像素长度: {ref_px:.1f} px")
            print(f"  比例: {self.pixels_per_cm:.2f} px/cm")
            print("  距离标定完成，开始坐姿检测")
            return True

        return False

    def get_distance_cm(self, nose_x, nose_y):
        """
        计算鼻子到屏幕参考点的实际距离

        Args:
            nose_x: 鼻子关键点的 x 坐标
            nose_y: 鼻子关键点的 y 坐标

        Returns:
            distance_cm: 实际距离（厘米），标定未完成或缺少数据返回 None
        """
        if not self.is_calibrated or self.hand_point is None:
            return None
        if self.pixels_per_cm is None or self.pixels_per_cm == 0:
            return None

        dist_px = self._pixel_distance((nose_x, nose_y), self.hand_point)
        return dist_px / self.pixels_per_cm

    def get_nose_keypoint(self, keypoints):
        """从关键点列表中提取鼻子坐标"""
        for kp in keypoints:
            if kp['index'] == 0:
                return (kp['x'], kp['y'])
        return None

    def draw_calib_info(self, image):
        """
        在图像上绘制标定状态和已有标注

        Args:
            image: BGR 格式的 numpy 数组

        Returns:
            output_image: 绘制后的图像
        """
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (0, 255, 255)

        # 绘制已标注的点
        if self.hand_point is not None:
            cv2.circle(output, self.hand_point, 6, (255, 0, 255), -1)
            cv2.circle(output, self.hand_point, 8, (255, 255, 255), 2)
            cv2.putText(output, "Screen", (self.hand_point[0] + 10, self.hand_point[1] - 5),
                        font, font_scale, (255, 0, 255), thickness)

        if self.ref_start is not None:
            cv2.circle(output, self.ref_start, 5, (255, 255, 0), -1)
            cv2.circle(output, self.ref_start, 7, (255, 255, 255), 2)

        if self.ref_end is not None:
            cv2.circle(output, self.ref_end, 5, (255, 255, 0), -1)
            cv2.circle(output, self.ref_end, 7, (255, 255, 255), 2)
            # 绘制参考线段
            if self.ref_start is not None:
                cv2.line(output, self.ref_start, self.ref_end, (255, 255, 0), 2)
                mid_x = (self.ref_start[0] + self.ref_end[0]) // 2
                mid_y = (self.ref_start[1] + self.ref_end[1]) // 2
                cv2.putText(output, "20cm", (mid_x + 5, mid_y - 5),
                            font, font_scale, (255, 255, 0), thickness)

        # 绘制标定提示
        if self.calib_state == self.CALIB_HAND:
            hint = "Click to mark screen reference point (1/3)"
        elif self.calib_state == self.CALIB_REF_START:
            hint = "Click start of 20cm reference line (2/3)"
        elif self.calib_state == self.CALIB_REF_END:
            hint = "Click end of 20cm reference line (3/3)"
        else:
            return output

        cv2.putText(output, hint, (20, 80), font, font_scale, color, thickness)
        return output

    def draw_distance_info(self, image, keypoints):
        """
        在图像上绘制实时距离信息

        Args:
            image: BGR 格式的 numpy 数组
            keypoints: 关键点列表

        Returns:
            output_image: 绘制后的图像
        """
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        nose = self.get_nose_keypoint(keypoints)
        if nose is None:
            cv2.putText(output, "Distance: N/A", (20, 40 + 35 * 3),
                        font, font_scale, (128, 128, 128), thickness)
            return output

        dist_cm = self.get_distance_cm(nose[0], nose[1])

        if dist_cm is not None:
            if dist_cm < 30:
                color = (0, 0, 255)          # 红色
            elif dist_cm < 40:
                color = (0, 165, 255)        # 橙色
            else:
                color = (0, 255, 0)          # 绿色

            cv2.putText(output, f"Distance: {dist_cm:.1f} cm", (20, 40 + 35 * 3),
                        font, font_scale, color, thickness)
        else:
            cv2.putText(output, "Distance: N/A", (20, 40 + 35 * 3),
                        font, font_scale, (128, 128, 128), thickness)

        return output

    @staticmethod
    def _pixel_distance(p1, p2):
        """计算两个像素点之间的欧氏距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
