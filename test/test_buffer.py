#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - GPIO 控制测试

功能：
1. 连接 GPIO17 控制蜂鸣器
2. 当 GPIO17 为高电平时，蜂鸣器发声
3. 提供简单的测试接口

使用说明：
python3 test/test_buffer.py

作者：30740
日期：2026 年 4 月 25 日
"""

import time
import argparse
import sys

# 尝试导入 gpiozero 库并初始化
try:
    from gpiozero import Buzzer
    # 尝试初始化一个临时 Buzzer 来测试 GPIO 可用性
    test_buzzer = Buzzer(17)
    test_buzzer.close()
    GPIO_AVAILABLE = True
except Exception as e:
    print(f"⚠️  GPIO 初始化失败: {e}")
    print("切换到模拟模式...")
    GPIO_AVAILABLE = False

# 模拟 Buzzer 类，用于在没有 gpiozero 库时提供占位符
class MockBuzzer:
    """模拟 Buzzer 类"""
    
    def __init__(self, pin):
        self.pin = pin
        print(f"📋 初始化模拟蜂鸣器，引脚: {pin}")
    
    def on(self):
        print("📋 模拟蜂鸣器: 开启")
    
    def off(self):
        print("📋 模拟蜂鸣器: 关闭")
    
    def beep(self, on_time=1, off_time=1, n=1, background=False):
        print(f"📋 模拟蜂鸣器: 发声 {n} 次，每次 {on_time} 秒")
        for _ in range(n):
            self.on()
            time.sleep(on_time)
            self.off()
            if _ < n - 1:
                time.sleep(off_time)
    
    def close(self):
        print("📋 模拟蜂鸣器: 关闭")

# 选择 Buzzer 实现
if not GPIO_AVAILABLE:
    Buzzer = MockBuzzer
    print("✅ 使用模拟 Buzzer 模式")
else:
    print("✅ gpiozero 库加载成功")


class BuzzerController:
    """蜂鸣器控制器"""
    
    def __init__(self, pin=17):
        """
        初始化蜂鸣器控制器
        
        Args:
            pin: GPIO 引脚号（BCM 编号）
        """
        self.pin = pin
        self.buzzer = Buzzer(pin)
        self.gpio_available = GPIO_AVAILABLE
        print(f"✅ 蜂鸣器控制器初始化完成，引脚: {pin}")
    
    def beep(self, duration=0.5):
        """
        蜂鸣器发声
        
        Args:
            duration: 发声持续时间（秒）
        """
        if self.gpio_available:
            print(f"🔊 蜂鸣器发声，持续 {duration} 秒")
            self.buzzer.on()
            time.sleep(duration)
            self.buzzer.off()
            print("🔇 蜂鸣器停止")
        else:
            print(f"� 模拟蜂鸣器发声，持续 {duration} 秒")
            self.buzzer.beep(on_time=duration, n=1)
    
    def set_state(self, state):
        """
        设置蜂鸣器状态
        
        Args:
            state: True 为开启（发声），False 为关闭（不发声）
        """
        if state:
            self.buzzer.on()
            if self.gpio_available:
                print("📡 蜂鸣器状态: 开启")
            else:
                print("� 模拟蜂鸣器状态: 开启")
        else:
            self.buzzer.off()
            if self.gpio_available:
                print("📡 蜂鸣器状态: 关闭")
            else:
                print("📋 模拟蜂鸣器状态: 关闭")
    
    def cleanup(self):
        """清理资源"""
        self.buzzer.off()
        self.buzzer.close()
        if self.gpio_available:
            print("✅ 蜂鸣器资源已清理")
        else:
            print("📋 模拟蜂鸣器资源已清理")


def test_buzzer():
    """测试蜂鸣器"""
    print("=" * 60)
    print("树莓派 5 坐姿检测系统 - 蜂鸣器测试")
    print("=" * 60)
    
    buzzer = BuzzerController(pin=17)
    
    try:
        print("\n测试 1: 短声")
        buzzer.beep(0.5)
        
        time.sleep(1)
        
        print("\n测试 2: 长声")
        buzzer.beep(2)
        
        time.sleep(1)
        
        print("\n测试 3: 手动控制")
        print("设置为开启...")
        buzzer.set_state(True)
        time.sleep(1)
        print("设置为关闭...")
        buzzer.set_state(False)
        
        print("\n✅ 蜂鸣器测试完成")
        
    finally:
        buzzer.cleanup()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='树莓派 5 蜂鸣器测试')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--beep', type=float, default=0.5, help='发声持续时间（秒）')
    
    args = parser.parse_args()
    
    if args.test:
        test_buzzer()
    else:
        buzzer = BuzzerController(pin=17)
        try:
            buzzer.beep(args.beep)
        finally:
            buzzer.cleanup()


if __name__ == "__main__":
    main()
