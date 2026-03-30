from AIDetector_pytorch import Detector
import imutils
import cv2


def main():
    func_status = {}
    func_status['headpose'] = None

    # 初始化检测器
    det = Detector()

    # ===================== 【全速配置】 =====================
    SKIP_FRAME = 2  # 跳帧（越大越快，1=不跳，2=跳1帧，3=跳2帧）
    PROCESS_HEIGHT = 416  # 处理尺寸（越小越快，320/416/512）
    VIDEO_OUTPUT = 'result.mp4'  # 输出视频名
    # =======================================================

    # 打开视频
    cap = cv2.VideoCapture(r"C:\Users\Chinasoft\Downloads\yolo-deepsort-inference.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频总帧数：{total_frames}, 原FPS：{fps}')

    videoWriter = None
    frame_count = 0

    print("开始后台处理...")

    while True:
        ret, im = cap.read()
        if not ret:
            break  # 视频读完结束

        frame_count += 1

        # ========== 跳帧加速（不处理的帧直接跳过） ==========
        if frame_count % SKIP_FRAME != 0:
            continue

        # ========== 缩小图像，大幅加速推理 ==========
        im = imutils.resize(im, height=PROCESS_HEIGHT)

        # ========== 目标检测 + 追踪 ==========
        result = det.feedCap(im, func_status)
        result = result['frame']

        # ========== 初始化视频写入器 ==========
        if videoWriter is None:
            h, w = result.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 输出视频 FPS 自动适配跳帧
            out_fps = max(10, fps // SKIP_FRAME)
            videoWriter = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, out_fps, (w, h))

        # ========== 写入视频（唯一操作） ==========
        videoWriter.write(result)

        # 打印进度（可选）
        print(f'处理进度：{frame_count}/{total_frames} 帧', end='\r')

    # 释放资源
    cap.release()
    if videoWriter is not None:
        videoWriter.release()

    print("\n处理完成！视频已保存为：", VIDEO_OUTPUT)


if __name__ == '__main__':
    main()