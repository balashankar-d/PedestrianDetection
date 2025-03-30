import cv2
import numpy as np

def visualize_fusion_pedestrians_only(image, results):
    if image is None:
        raise ValueError("The image could not be loaded. Please check the file path.")

    for result in results: 
        if not hasattr(result, "masks") or result.masks is None or result.masks.data is None:
            continue

        for box, mask in zip(result.boxes, result.masks.data):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            cls = int(box.cls[0])
            if result.names[cls] != "person":
                continue

            label = f"{result.names[cls]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Process mask
            mask_array = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_array, (image.shape[1], image.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            mask_color = np.zeros_like(image, dtype=np.uint8)
            mask_color[:, :, 1] = mask_binary * 255

            # Apply mask overlay
            image = cv2.addWeighted(image, 1.0, mask_color, 0.5, 0)

    # ⚠️ **REMOVED plt.show() TO AVOID BLOCKING REAL-TIME DISPLAY**

    # Show processed frame in OpenCV window
    cv2.imshow("Pedestrian Detection", image)
