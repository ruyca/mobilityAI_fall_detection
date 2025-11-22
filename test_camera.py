import cv2

cap = cv2.VideoCapture(0)  # 0 = first USB camera

if not cap.isOpened():
    print("Camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("USB Camera", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
