from picamera2 import Picamera2

def test_camera():
    # 初始化相机
    picam2 = Picamera2()
    
    # 配置相机为高分辨率
    config = picam2.create_still_configuration(main={"size": (1280, 960)})
    picam2.configure(config)
    
    # 启动相机
    picam2.start()
    print("摄像头已启动...")
    print("分辨率设置为: 1280x960")
        
    # 捕获一张高分辨率照片并保存
    picam2.capture_file("test_photo.jpg")
    print("照片已保存为 test_photo.jpg")
    picam2.stop()
    picam2.close()

if __name__ == "__main__":
    test_camera()