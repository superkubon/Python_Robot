import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print("Kích thước frame:", frame.shape)  # In ra (h, w, c)
    flipped = cv2.flip(frame, 0)  # 1: horizontal flip
    cv2.imshow('Flipped Camera', flipped)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
