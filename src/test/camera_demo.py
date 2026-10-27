import os
import cv2
import torch
from ultralytics import YOLO

# 1. Cấu hình cờ FFmpeg xóa buffer để giảm trễ cho IP Camera
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

# 2. Danh sách 17 Keypoints theo chuẩn COCO
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 3. Load mô hình YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Tự động chọn GPU nếu có, ngược lại dùng CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy mô hình trên thiết bị: {device.upper()}")

# 4. Kết nối tới luồng IP Camera từ điện thoại
stream_url = "http://53.72.132.202:8080/video"
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

# Giới hạn buffer về 1 khung hình để luôn lấy frame mới nhất
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"Không thể kết nối tới camera tại {stream_url}. Kiểm tra lại Wi-Fi/Địa chỉ IP!")
    exit()

print("Kết nối thành công! Nhấn 'q' trên cửa sổ hình ảnh để thoát.")

# 5. Cấu hình cửa sổ hiển thị linh hoạt (cho phép co giãn, thu nhỏ)
window_name = "ZeroTouch Care - Live Pose Estimation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 450, 800)  # Đặt kích thước hiển thị vừa mắt

# 6. Vòng lặp xử lý khung hình thời gian thực
while True:
    ret, frame = cap.read()
    if not ret:
        print("Mất tín hiệu từ luồng camera...")
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Trích xuất Keypoints bằng YOLOv8 Pose
    results = model.predict(
        source=frame,
        device=device,
        stream=True,
        verbose=False,
        conf=0.5
    )

    for result in results:
        # Kiểm tra nếu phát hiện người trong khung hình
        if result.keypoints is not None and len(result.keypoints) > 0:
            xy_coords = result.keypoints.xy    # Tensor dạng (num_persons, 17, 2)
            conf_scores = result.keypoints.conf  # Tensor dạng (num_persons, 17)

            # --- VÙNG XỬ LÝ DỰ ĐOÁN NGÃ (VÍ DỤ TRUY CẬP KEYPOINTS) ---
            # Lấy tọa độ của người đầu tiên phát hiện được
            # person_0_xy = xy_coords[0] 
            # shoulder_y = person_0_xy[5][1].item()  # Y-coord của Left Shoulder
            # hip_y = person_0_xy[11][1].item()       # Y-coord của Left Hip
            # -------------------------------------------------------

        # Vẽ Bounding Box và Skeleton khớp xương lên frame
        annotated_frame = result.plot()

    # Hiển thị kết quả ra cửa sổ đã cấu hình
    cv2.imshow(window_name, annotated_frame)

    # Nhấn 'q' để dừng chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()