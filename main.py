import cv2
from ultralytics import YOLO
from utils import visualize_fusion_pedestrians_only

def main():
   
    model = YOLO('yolov8s-seg.pt')  
    image_path = "input_images/pedestrain.jpg" 
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image. Check the file path.")
        return

    results = model(image)
    visualize_fusion_pedestrians_only(image, results)

if __name__ == "__main__":
    main()
