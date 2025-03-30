import cv2
import ultralytics
from ultralytics import YOLO
from utils import visualize_fusion_pedestrians_only

def main():
    model = YOLO('yolov8s-seg.pt')  
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break  

        # Run YOLO model on the frame
        results = model(frame)

        # Visualize results in OpenCV (no matplotlib blocking)
        visualize_fusion_pedestrians_only(frame, results)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
