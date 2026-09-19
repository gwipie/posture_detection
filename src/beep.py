#!/usr/bin/env python3
"""
树莓派 5 坐姿检测系统 - 蜂鸣器报警模块

功能：
1. 根据坐姿偏差指标控制蜂鸣器报警
2. 橙色阈值持续20秒后，蜂鸣器以1次/秒频率发声
3. 红色阈值持续10秒后，蜂鸣器以2次/秒频率发声
4. 橙色跳到红色时，橙色计时器清零，红色计时器开始计时
5. 蜂鸣器发声后，直到全部指标恢复正常才停止

使用说明：
该模块由 main.py 调用，不单独运行

作者：30740
日期：2026 年 4 月 25 日
"""

import time

try:
    from gpiozero import Buzzer as _RealBuzzer
    _test = _RealBuzzer(17)
    _test.close()
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


class _MockBuzzer:
    def __init__(self, pin):
        self.pin = pin

    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass


_BuzzerClass = _RealBuzzer if GPIO_AVAILABLE else _MockBuzzer


class PostureAlerter:
    """坐姿报警控制器"""

    ALERT_NONE = 0
    ALERT_ORANGE = 1
    ALERT_RED = 2

    def __init__(self, pin=17,
                 head_orange_threshold=10, head_red_threshold=20,
                 body_orange_threshold=10, body_red_threshold=20,
                 back_orange_threshold=12, back_red_threshold=20,
                 orange_duration=20.0, red_duration=10.0):
        """
        初始化坐姿报警控制器

        Args:
            pin: 蜂鸣器 GPIO 引脚号（BCM 编号）
            head_orange_threshold: 头部倾角橙色阈值（度）
            head_red_threshold: 头部倾角红色阈值（度）
            body_orange_threshold: 身体倾角橙色阈值（度）
            body_red_threshold: 身体倾角红色阈值（度）
            back_orange_threshold: 背部弯曲橙色阈值（像素）
            back_red_threshold: 背部弯曲红色阈值（像素）
            orange_duration: 橙色状态触发报警的持续时间（秒）
            red_duration: 红色状态触发报警的持续时间（秒）
        """
        self.buzzer = _BuzzerClass(pin)
        self.gpio_available = GPIO_AVAILABLE

        self.thresholds = {
            'head': (head_orange_threshold, head_red_threshold),
            'body': (body_orange_threshold, body_red_threshold),
            'back': (back_orange_threshold, back_red_threshold),
        }
        self.orange_duration = orange_duration
        self.red_duration = red_duration

        self.metrics = {}
        for name in ('head', 'body', 'back'):
            self.metrics[name] = {
                'state': 'green',
                'orange_start': None,
                'red_start': None,
            }

        self.is_alerting = False
        self.alert_level = self.ALERT_NONE
        self.beep_cycle_start = 0
        self.beep_on = False
        self.beep_on_duration = 0.2

    def _get_state(self, name, value):
        """判断单个指标的当前状态"""
        if value is None:
            return 'unknown'

        orange_th, red_th = self.thresholds[name]
        abs_val = abs(value) if name in ('head', 'body') else value

        if abs_val >= red_th:
            return 'red'
        elif abs_val >= orange_th:
            return 'orange'
        return 'green'

    def _update_metric(self, name, value):
        """更新单个指标的状态和计时器"""
        new_state = self._get_state(name, value)
        m = self.metrics[name]
        old_state = m['state']

        if new_state == old_state:
            return

        now = time.time()

        if new_state == 'orange':
            m['orange_start'] = now
            m['red_start'] = None
        elif new_state == 'red':
            m['orange_start'] = None
            m['red_start'] = now
        else:  # green or unknown: do not accumulate an alert without valid keypoints
            m['orange_start'] = None
            m['red_start'] = None

        m['state'] = new_state

    def _check_alerts(self):
        """检查是否有报警条件满足，返回最高报警级别"""
        now = time.time()
        has_orange = False
        has_red = False

        for m in self.metrics.values():
            if m['state'] == 'red' and m['red_start'] is not None:
                if now - m['red_start'] >= self.red_duration:
                    has_red = True
            if m['state'] == 'orange' and m['orange_start'] is not None:
                if now - m['orange_start'] >= self.orange_duration:
                    has_orange = True

        if has_red:
            return self.ALERT_RED
        if has_orange:
            return self.ALERT_ORANGE
        return self.ALERT_NONE

    def _all_green(self):
        """检查是否所有指标都恢复正常"""
        return all(m['state'] == 'green' for m in self.metrics.values())

    def _has_unknown(self):
        """检查是否有指标因关键点缺失而无法判断。"""
        return any(m['state'] == 'unknown' for m in self.metrics.values())

    def _stop_alert(self):
        self.is_alerting = False
        self.alert_level = self.ALERT_NONE
        self.buzzer.off()
        self.beep_on = False
        self.beep_cycle_start = 0

    def _handle_beep(self):
        """根据报警级别控制蜂鸣器的发声模式"""
        now = time.time()

        if self.alert_level == self.ALERT_RED:
            period = 0.5
        else:
            period = 1.0

        if self.beep_cycle_start == 0:
            self.beep_cycle_start = now

        elapsed = now - self.beep_cycle_start
        cycle_pos = elapsed % period

        if cycle_pos < self.beep_on_duration:
            if not self.beep_on:
                self.buzzer.on()
                self.beep_on = True
        else:
            if self.beep_on:
                self.buzzer.off()
                self.beep_on = False

    def update(self, head_diff, body_diff, back_diff):
        """
        每帧调用，更新报警状态

        Args:
            head_diff: 头部倾角偏差（度）
            body_diff: 身体倾角偏差（度）
            back_diff: 背部弯曲偏差（像素）
        """
        self._update_metric('head', head_diff)
        self._update_metric('body', body_diff)
        self._update_metric('back', back_diff)

        current_alert = self._check_alerts()

        if current_alert > self.ALERT_NONE:
            if not self.is_alerting:
                self.is_alerting = True
                self.beep_cycle_start = 0
                self.beep_on = False
            if current_alert != self.alert_level:
                self.alert_level = current_alert
                self.beep_cycle_start = 0
                self.beep_on = False
        elif self._has_unknown():
            self._stop_alert()
        else:
            if self.is_alerting and self._all_green():
                self._stop_alert()
            elif self.is_alerting:
                self.alert_level = self.ALERT_ORANGE

        if self.is_alerting:
            self._handle_beep()

    def reset(self):
        """重置所有状态（重新记录基准时调用）"""
        for m in self.metrics.values():
            m['state'] = 'green'
            m['orange_start'] = None
            m['red_start'] = None
        self._stop_alert()

    def cleanup(self):
        """清理资源"""
        self.buzzer.off()
        self.buzzer.close()
