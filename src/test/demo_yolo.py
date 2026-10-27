import cv2
import torch
from ultralytics import YOLO

# 17 COCO Keypoint Class Names
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 1. Load the pre-trained YOLOv8 Pose model (COCO 17 keypoints)
model = YOLO("yolov8n-pose.pt")

# 2. Run inference on an image (replace 'path/to/image.jpg' with your image)
image_path = r"C:\Users\Viet\Desktop\1790563970926_6802166001166294531_g1365389796172088983_98472cd27e079e2a4ab7f1fa591cc56e.jpg"

results = model(image_path)

# 3. Process detection results
for result in results:
    # Check if any persons with keypoints were detected
    if result.keypoints is not None:
        keypoints = result.keypoints  # Keypoints object
        
        # Tensor of shape (num_persons, 17, 2) containing (x, y) coordinates
        xy_coords = keypoints.xy
        
        # Tensor of shape (num_persons, 17) containing confidence scores
        conf_scores = keypoints.conf

        for person_idx, (person_xy, person_conf) in enumerate(zip(xy_coords, conf_scores)):
            print(f"\n--- Person {person_idx + 1} ---")
            
            for kp_idx, (coord, conf) in enumerate(zip(person_xy, person_conf)):
                x, y = coord[0].item(), coord[1].item()
                c = conf.item()
                kp_name = COCO_KEYPOINTS[kp_idx]
                
                print(f"{kp_idx:2d} | {kp_name:<14} | X: {x:7.2f}, Y: {y:7.2f} | Confidence: {c:.2f}")

    # 4. Save visual output with keypoints and skeleton drawn
    annotated_img = result.plot()
    cv2.imwrite("pose_output.jpg", annotated_img)
    print("\nAnnotated result saved as 'pose_output.jpg'.")