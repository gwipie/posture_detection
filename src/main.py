#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - 主程序

功能：
1. 实时调用摄像头，在屏幕上输出视频
2. 将摄像头的 1280x960 画面转换为 640x640 输入到模型中
3. 模型检测人体的关键点：鼻子、右肩、右髋、右膝
4. 在屏幕上显示模型的标注，保持 1280x960 的显示尺寸
5. 记录标准坐姿角度，实时显示角度偏差
6. 标定头部到屏幕距离，实时显示距离信息

使用说明：
python3 src/main.py
- 程序启动后，请保持端正坐姿，按 's' 键记录标准坐姿
- 记录后进入距离标定模式，按提示依次点击屏幕参考点和20cm参考线段
- 标定完成后进入实时检测模式，屏幕上显示角度偏差和距离
- 按 'q' 键退出

作者：30740
日期：2026 年 4 月 26 日
"""

import os
import sys
from pathlib import Path
from picamera2 import Picamera2
import cv2

from posture_detection import PoseDetector, PoseVisualizer
from record_info import PostureRecorder
from beep import PostureAlerter
from distance_detection import DistanceDetector


def draw_angle_info(image, recorder, keypoints):
    """
    在图像上绘制角度偏差信息

    Args:
        image: BGR 格式的 numpy 数组
        recorder: PostureRecorder 实例
        keypoints: 关键点列表

    Returns:
        output_image: 绘制后的图像
    """
    output_image = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    x_offset = 20
    y_offset = 40
    line_height = 35

    if not recorder.is_baseline_set:
        cv2.putText(
            output_image,
            "Press 's' to record baseline posture",
            (x_offset, y_offset),
            font, font_scale, (0, 255, 255), thickness
        )
        return output_image

    head_diff, body_diff, back_diff = recorder.get_angle_diffs(keypoints)

    head_color = (0, 255, 0)
    body_color = (0, 255, 0)
    back_color = (0, 255, 0)
    # 头部倾角偏差
    if head_diff is not None:
        if abs(head_diff) > 20:
            head_color = (0, 0, 255)
        elif abs(head_diff) > 10:
            head_color = (0, 165, 255)

        cv2.putText(
            output_image,
            f"Head tilt diff: {head_diff:+.1f} deg",
            (x_offset, y_offset),
            font, font_scale, head_color, thickness
        )
    else:
        cv2.putText(
            output_image,
            "Head tilt diff: N/A",
            (x_offset, y_offset),
            font, font_scale, (128, 128, 128), thickness
        )
    # 身体倾角偏差
    if body_diff is not None:
        if abs(body_diff) > 20:
            body_color = (0, 0, 255)
        elif abs(body_diff) > 10:
            body_color = (0, 165, 255)

        cv2.putText(
            output_image,
            f"Body tilt diff: {body_diff:+.1f} deg",
            (x_offset, y_offset + line_height),
            font, font_scale, body_color, thickness
        )
    else:
        cv2.putText(
            output_image,
            "Body tilt diff: N/A",
            (x_offset, y_offset + line_height),
            font, font_scale, (128, 128, 128), thickness
        )
    # 背部弯曲角度偏差
    if back_diff is not None:
        if back_diff > 20:
            back_color = (0, 0, 255)
        elif back_diff > 10:
            back_color = (0, 165, 255)

        cv2.putText(
            output_image,
            f"Back bend diff: {back_diff:+.1f} px",
            (x_offset, y_offset + line_height * 2),
            font, font_scale, back_color, thickness
        )
    else:
        cv2.putText(
            output_image,
            "Back bend diff: N/A",
            (x_offset, y_offset + line_height * 2),
            font, font_scale, (128, 128, 128), thickness
        )

    return output_image


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    model_path = project_root / 'model' / 'yolo11n-pose_int8.onnx'

    print("=" * 60)
    print("树莓派 5 坐姿检测系统 - 实时检测")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        sys.exit(1)

    print("📷 初始化摄像头...")
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(main={"size": (1280, 960)})
    picam2.configure(config)

    picam2.start()
    print("  ✅ 摄像头已启动")
    print("  摄像头分辨率：1280x960")

    print(f"\n🤖 加载模型：{model_path}")
    detector = PoseDetector(str(model_path))

    visualizer = PoseVisualizer()
    recorder = PostureRecorder()
    alerter = PostureAlerter(pin=17)
    dist_detector = DistanceDetector()

    # 运行阶段: 'setup' -> 'calibration' -> 'detection'
    phase = 'setup'

    print("\n📋 操作说明：")
    print("  按 's' 键记录标准坐姿")
    print("  按 'q' 键退出")

    print("\n🔍 等待记录标准坐姿...")

    # 鼠标回调（仅标定阶段生效）
    def on_mouse(event, x, y, flags, param):
        nonlocal phase
        if event == cv2.EVENT_LBUTTONDOWN and phase == 'calibration':
            calib_done = dist_detector.handle_click(x, y)
            if calib_done:
                phase = 'detection'
                print("\n🔍 开始实时坐姿检测...")

    try:
        while True:
            frame = picam2.capture_array()

            keypoints = detector.detect(frame)

            result_frame = visualizer.draw_pose(frame, keypoints)

            if phase == 'setup':
                result_frame = draw_angle_info(result_frame, recorder, keypoints)

            elif phase == 'calibration':
                result_frame = dist_detector.draw_calib_info(result_frame)

            elif phase == 'detection':
                result_frame = draw_angle_info(result_frame, recorder, keypoints)
                result_frame = dist_detector.draw_distance_info(result_frame, keypoints)

                head_diff, body_diff, back_diff = recorder.get_angle_diffs(keypoints)
                alerter.update(head_diff, body_diff, back_diff)

            cv2.imshow('Posture Detection', result_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s') and phase == 'setup':
                print("\n📝 正在记录标准坐姿...")
                if recorder.record_baseline(keypoints):
                    alerter.reset()
                    print("  进入距离标定模式")
                    phase = 'calibration'
                    dist_detector.start_calibration()
                    cv2.setMouseCallback('Posture Detection', on_mouse)
                else:
                    print("  ❌ 记录失败，请确保关键点可见后重试")

    finally:
        cv2.setMouseCallback('Posture Detection', lambda *a: None)
        alerter.cleanup()
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("\n✅ 检测已停止")


if __name__ == "__main__":
    main()
