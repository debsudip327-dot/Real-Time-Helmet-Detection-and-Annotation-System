import cv2
from ultralytics import YOLO

model = YOLO(r'runs\detect\train-6\weights\best.pt')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.50, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        
        if cls_id == 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        if cls_id == 1:
            display_label = "No Helmet"
            color = (0, 0, 255)  # Red
        elif cls_id == 2:
            display_label = "Helmet"
            color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{display_label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Real-Time Helmet Detection (30 Epochs)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()