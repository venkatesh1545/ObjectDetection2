import cv2
from ultralytics import YOLO

# Load your model (choose the best for your hardware)
model = YOLO("yolo11n.pt")

# Open your webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    # Run detection
    results = model(frame)
    # Annotate results on frame
    annotated_frame = results[0].plot()
    # Show frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
