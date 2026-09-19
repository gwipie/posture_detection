#!/usr/bin/env python3
"""
树莓派5坐姿检测系统 - 模型验证脚本

功能：
1. 验证yolo11n-pose_int8.onnx模型在树莓派5上的兼容性
2. 测试模型推理性能和内存占用
3. 检查输入输出格式

使用说明：
在树莓派5上运行：
python3 test_model.py --model path/to/yolo11n-pose_int8.onnx

作者：30740
日期：2026年4月21日
"""

import argparse
import time
import os
import sys
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    import psutil
    import cv2
except ImportError as e:
    print(f"❌ 依赖库缺失: {e}")
    print("请先安装依赖：")
    print("  pip install onnxruntime numpy opencv-python psutil")
    sys.exit(1)


def check_dependencies():
    """检查所有依赖库是否安装正确"""
    print("🔍 检查依赖库...")
    
    # 检查ONNX Runtime版本
    try:
        print(f"  ✓ ONNX Runtime: {ort.__version__}")
        # 检查支持的providers
        available_providers = ort.get_available_providers()
        print(f"  ✓ 可用推理后端: {available_providers}")
    except Exception as e:
        print(f"  ❌ ONNX Runtime检查失败: {e}")
        return False
    
    # 检查OpenCV
    try:
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"  ❌ OpenCV检查失败: {e}")
        return False
    
    # 检查NumPy
    try:
        print(f"  ✓ NumPy: {np.__version__}")
    except Exception as e:
        print(f"  ❌ NumPy检查失败: {e}")
        return False
    
    return True


def validate_model(model_path):
    """
    验证ONNX模型文件
    
    Args:
        model_path: ONNX模型文件路径
        
    Returns:
        tuple: (success, session, input_info, output_info)
    """
    print(f"🔧 验证模型: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"  ❌ 模型文件不存在: {model_path}")
        return False, None, None, None
    
    # 检查文件大小
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"  ✓ 模型大小: {file_size:.1f} MB")
    
    # 加载模型
    try:
        # 使用CPU推理（树莓派5默认）
        providers = ['CPUExecutionProvider']
        
        # 创建推理会话
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # 减少日志输出
        
        session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options
        )
        print("  ✓ 模型加载成功")
        
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return False, None, None, None
    
    # 获取输入输出信息
    try:
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"  ✓ 输入名称: {input_info.name}")
        print(f"  ✓ 输入形状: {input_info.shape}")
        print(f"  ✓ 输入类型: {input_info.type}")
        print(f"  ✓ 输出数量: {len(output_info)}")
        
        for i, out in enumerate(output_info):
            print(f"    - 输出{i}: {out.name}, 形状: {out.shape}, 类型: {out.type}")
            
    except Exception as e:
        print(f"  ❌ 获取模型信息失败: {e}")
        return False, None, None, None
    
    return True, session, input_info, output_info


def benchmark_inference(session, input_info, output_info, iterations=100):
    """
    基准测试推理性能
    
    Args:
        session: ONNX Runtime会话
        input_info: 输入信息
        output_info: 输出信息
        iterations: 测试迭代次数
    """
    print(f"⚡ 性能基准测试 ({iterations}次迭代)...")
    
    # 解析输入形状
    # 典型形状: [batch_size, channels, height, width]
    input_shape = input_info.shape
    batch_size = input_shape[0] if input_shape[0] > 0 else 1
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    
    print(f"  📐 输入尺寸: {batch_size}x{channels}x{height}x{width}")
    print(f"  📸 图像尺寸: {width}x{height}")
    
    # 准备测试输入数据
    # 使用随机数据（避免读取实际图像文件的依赖）
    test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    # 获取输入名称
    input_name = input_info.name
    
    # 预热（先运行几次）
    print("  🔥 预热推理...")
    for _ in range(20):
        session.run(None, {input_name: test_input})
    
    # 记录内存使用
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss / 1024 / 1024  # MB
    
    # 性能测试
    inference_times = []
    
    print(f"  🏃 开始{iterations}次推理测试...")
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # 执行推理
        outputs = session.run(None, {input_name: test_input})
        peak_rss = max(peak_rss, process.memory_info().rss / 1024 / 1024)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        inference_times.append(inference_time)
        
        # 每5次打印进度
        if (i + 1) % 5 == 0:
            print(f"    已完成 {i+1}/{iterations} 次")
    
    # 计算结果
    avg_time = np.mean(inference_times)
    p50_time = np.percentile(inference_times, 50)
    p95_time = np.percentile(inference_times, 95)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    fps = 1000 / avg_time  # 帧率
    
    print("\n📊 性能测试结果:")
    print(f"  ⏱️  平均推理时间: {avg_time:.1f} ms")
    print(f"  📍 P50 / P95: {p50_time:.1f} / {p95_time:.1f} ms")
    print(f"  📉 最小推理时间: {min_time:.1f} ms")
    print(f"  📈 最大推理时间: {max_time:.1f} ms")
    print(f"  📊 时间标准差: {std_time:.1f} ms")
    print(f"  🎞️  理论帧率: {fps:.1f} FPS")
    print(f"  💾 进程 RSS 峰值: {peak_rss:.1f} MB")
    
    # 检查输出结构
    if outputs:
        print(f"\n🔍 输出结构分析:")
        for i, output in enumerate(outputs):
            print(f"  输出{i}: 形状={output.shape}, 类型={output.dtype}")
            
            # 如果是YOLO输出，通常包含检测结果
            if len(output.shape) == 3:
                print(f"    检测数量: {output.shape[1]}")
                print(f"    特征维度: {output.shape[2]}")
    
    return avg_time, p50_time, p95_time, fps, peak_rss


def test_with_sample_image(session, input_info, sample_path=None):
    """
    使用样本图像测试模型
    
    Args:
        session: ONNX Runtime会话
        input_info: 输入信息
        sample_path: 样本图像路径（可选）
    """
    print("\n🖼️  样本图像测试...")
    
    if sample_path and os.path.exists(sample_path):
        print(f"  使用样本图像: {sample_path}")
        # 读取并处理图像
        img = cv2.imread(sample_path)
        if img is None:
            print("  ⚠️  无法读取样本图像，使用随机数据继续测试")
            return False
    else:
        print("  ⚠️  未提供样本图像，使用随机数据测试")
        return False
    
    # 获取输入形状
    input_shape = input_info.shape
    batch_size = input_shape[0] if input_shape[0] > 0 else 1
    channels = input_shape[1]
    target_height = input_shape[2]
    target_width = input_shape[3]
    
    # 调整图像尺寸
    img_resized = cv2.resize(img, (target_width, target_height))
    
    # 转换通道顺序: BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 归一化 (如果需要)
    # YOLO通常需要归一化到0-1范围
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # 调整维度顺序: HWC -> NCHW
    img_input = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)       # CHW to NCHW
    
    # 执行推理
    input_name = input_info.name
    try:
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: img_input})
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000
        print(f"  ✅ 样本图像推理成功")
        print(f"  ⏱️  推理时间: {inference_time:.1f} ms")
        
        # 简单解析输出
        if outputs and len(outputs) > 0:
            output = outputs[0]
            if len(output.shape) == 3:
                detections = output[0]  # 第一维通常是batch
                print(f"  🔍 检测到 {len(detections)} 个候选结果")
                
                # 显示前几个检测结果
                for i, det in enumerate(detections[:3]):
                    if len(det) >= 6:  # 通常包含xywh, confidence, class
                        print(f"    检测{i}: 置信度={det[4]:.3f}, 类别={int(det[5])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 样本图像推理失败: {e}")
        return False


def main():
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    default_model_path = project_root / 'model' / 'yolo11n-pose_int8.onnx'
    
    parser = argparse.ArgumentParser(description='验证 YOLOv11n-pose 模型在树莓派 5 上的兼容性')
    parser.add_argument('--model', type=str, default=str(default_model_path),
                       help=f'ONNX 模型文件路径 (默认：{default_model_path})')
    parser.add_argument('--iterations', type=int, default=100,
                       help='性能测试迭代次数 (默认：100)')
    parser.add_argument('--sample', type=str, default=None,
                       help='样本图像路径 (可选)')
    parser.add_argument('--output', type=str, default='model_validation_report.txt',
                       help='输出报告文件路径 (默认：model_validation_report.txt)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("树莓派5坐姿检测系统 - 模型验证脚本")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请先安装必要的依赖库")
        sys.exit(1)
    
    # 验证模型
    success, session, input_info, output_info = validate_model(args.model)
    if not success:
        print("❌ 模型验证失败")
        sys.exit(1)
    
    # 性能基准测试
    avg_time, p50_time, p95_time, fps, peak_rss = benchmark_inference(
        session, input_info, output_info, args.iterations
    )
    
    # 样本图像测试（可选）
    if args.sample:
        test_with_sample_image(session, input_info, args.sample)
    
    # 生成报告
    report = generate_report(args.model, avg_time, p50_time, p95_time, fps, peak_rss, input_info)
    save_report(report, args.output)
    
    print("\n" + "=" * 60)
    print("✅ 模型验证完成")
    print("=" * 60)
    
    # 给出建议
    print(f"纯模型推理吞吐估算: {fps:.1f} FPS；是否满足实时要求需结合端到端时延判断。")
    
    print(f"\n📄 详细报告已保存到: {args.output}")


def generate_report(model_path, avg_time, p50_time, p95_time, fps, peak_rss, input_info):
    """生成验证报告"""
    import platform
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("树莓派5坐姿检测系统 - 模型验证报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"系统: {platform.system()} {platform.release()}")
    report_lines.append(f"Python版本: {platform.python_version()}")
    report_lines.append("")
    
    report_lines.append("📁 模型信息:")
    report_lines.append(f"  模型路径: {model_path}")
    report_lines.append(f"  模型大小: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    report_lines.append(f"  输入形状: {input_info.shape}")
    report_lines.append(f"  输入类型: {input_info.type}")
    report_lines.append("")
    
    report_lines.append("📊 性能测试结果:")
    report_lines.append(f"  平均推理时间: {avg_time:.1f} ms")
    report_lines.append(f"  P50 / P95 推理时间: {p50_time:.1f} / {p95_time:.1f} ms")
    report_lines.append(f"  纯推理吞吐估算: {fps:.1f} FPS")
    report_lines.append(f"  进程 RSS 峰值: {peak_rss:.1f} MB")
    report_lines.append("")
    
    report_lines.append("🎯 口径说明:")
    report_lines.append("  以上只测模型推理，不包含摄像头、预处理、后处理、显示和GPIO。")
    report_lines.append("  是否满足实时要求应由业务时延预算和端到端基准共同判断。")
    report_lines.append("")
    
    report_lines.append("💡 优化建议:")
    report_lines.append("  1. 确保使用INT8量化模型以获得最佳性能")
    report_lines.append("  2. 调整输入分辨率以平衡速度与精度")
    report_lines.append("  3. 分别测量预处理、推理、后处理和端到端 P50/P95")
    report_lines.append("  4. 监控树莓派5温度以防止过热降频")
    report_lines.append("")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def save_report(report, output_path):
    """保存报告到文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ 报告已保存到: {output_path}")
    except Exception as e:
        print(f"⚠️  无法保存报告: {e}")


if __name__ == "__main__":
    main()
