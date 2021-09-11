import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(
    r"E:\Users\Bart\Documents\GitHub\pws\haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cropped_image = 0
resized = 0

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x - 5, y - 5),
                      (x + w + 5, y + h + 5), (255, 0, 0), 3)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        cropped_image = gray[y: y + h, x: x + w]
        resized = cv2.resize(cropped_image, [100, 100])

    cv2.imshow("img", img)
    cv2.imshow("img2", resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
