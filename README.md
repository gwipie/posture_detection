# posture_detection
基于树莓派5和YOLOv11_pose的人体坐姿检测系统
## 安装系统级依赖
在树莓派5上，官方推荐的Python相机库是 Picamera2。由于它依赖底层的C++库，最稳定的方式是在系统层安装，然后让虚拟环境继承。gpiozero 库也需要在系统层安装，系统应该已经包含了这个库。
```bash
sudo apt update
sudo apt install python3-picamera2 python3-libcamera python3-opencv python3-gpiozero
```
## 正确创建虚拟环境
普通的虚拟环境（python -m venv myenv）是完全隔离的，这会导致你无法导入系统级的 picamera2，从而调用失败。
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
## 安装项目所需依赖
不要在虚拟环境中安装 numpy和opencv-python，因为它会与系统级的 numpy、opencv-python 冲突。
```bash
pip install onnxruntime psutil   
```

## 功能简述
- 该系统基于树莓派5和YOLOv11_pose的人体坐姿检测系统。
- 系统通过Picamera2库获取视频流，使用YOLOv11_pose模型进行人体坐姿检测。
- 程序运行开始时，检测者端正坐姿，按 s 后记录正确坐姿的头部倾角、身体倾角和背部弯曲角度（通过肩髋线段的长度来反映）。
- 记录坐姿之后，进入距离标定阶段，检测者端正坐姿，进行手动距离标定。共三个标定点，分别是检测目标点，20cm线段的起点和终点。
- 程序运行时，系统不断检测人体的坐姿，然后显示与正确坐姿的对比结果。
- 当检测到人体的坐姿与正确坐姿的对比结果超出警告阈值20s后，系统会通过蜂鸣器警告（1t/s）检测者。
- 当检测到人体的坐姿与正确坐姿的对比结果超出危险阈值10s后，系统会通过蜂鸣器警告（2t/s）检测者。

## 使用边界

- 当前版本是单人坐姿原型：从同一个最高置信度人体候选中提取关键点，不支持多人身份保持。
- 关键点缺失时状态为 `unknown`，不会把“无法判断”当作正常坐姿，也不会继续累计提醒时间。
- 手动距离标定是二维近似；相机移动、透视变化或人与参考物不在同一平面时需要重新标定。
- `model_validation_report.txt` 是历史记录。请运行下面的命令生成当前环境报告，纯推理FPS不等于端到端FPS。

```bash
python test/test_model.py --iterations 100 --output model_validation_report_v2.txt
```
