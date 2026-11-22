import cv2


cap = cv2.VideoCapture("input.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video", frame)

    
    key = cv2.waitKey(50) & 0xFF

    if key == ord('s'):
        cv2.waitKey(200)

    
    elif key == ord('f'):
        cv2.waitKey(2)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()