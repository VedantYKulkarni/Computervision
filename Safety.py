import cv2
import torch

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Load the video stream (replace '0' with the video file path if using a video)
cap = cv2.VideoCapture(0)

# Define the Region of Interest (ROI) coordinates (x, y, width, height)
roi_x, roi_y, roi_width, roi_height = 200, 50, 400, 400

while True:
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        break

    # Perform hand detection with YOLOv5
    results = model(frame)

    # Get hand detections from the results
    hands = results.pred[0][(results.pred[0][:, -1] == 0) & (results.pred[0][:, -2] > 0.5)]
    

    # Draw bounding boxes for hands detected within ROI
    for hand in hands:
        x, y, w, h = hand[0:4].cpu().numpy()
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        

        # Check if the detected hand is within the ROI
        if roi_x <= x1 <= roi_x + roi_width and roi_y <= y1 <= roi_y + roi_height:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the ROI rectangle
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Hand Detection within ROI", frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video stream and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
