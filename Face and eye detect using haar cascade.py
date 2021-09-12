import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import winsound


face_cascade = cv2.CascadeClassifier(
    r"E:\Users\Bart\Documents\GitHub\pws\haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cropped_image = 0
resized = 0

model_1 = keras.models.load_model("model_1")

avPredictionStep = 0
avPredictionPerc = 0
avSize = 10
avPrediction = np.full(avSize, 50)


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 0:
        x, y, w, h = 0, 0, 5, 5
        for (xn, yn, wn, hn) in faces:
            if xn > x:
                x = xn
                y = yn
                w = wn
                h = hn

        cv2.rectangle(img, (x - 5, y - 5),
                      (x + w + 5, y + h + 5), (255, 0, 0), 3)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        cropped_image = gray[y: y + h, x: x + w]
        resized = cv2.resize(cropped_image, [100, 100])

    #
    test_image = (resized[..., ::-1].astype(np.float32)) / 255.0
    img_array = np.expand_dims(test_image, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = int(float(str(model_1.predict(img_array))[2:-2])*100)
    avPrediction[avPredictionStep] = prediction
    avPredictionPerc = np.average(avPrediction)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, str(avPredictionPerc), (10, 450), font,
                3, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("img", img)
    cv2.imshow("img2", resized)

    if avPredictionStep < avSize-1:
        avPredictionStep += 1
    else:
        avPredictionStep = 0

    if avPredictionPerc < 10:
        winsound.Beep(40, 100)
        #winsound.PlaySound("SystemExit", winsound.SND_LOOP | winsound.SND_ASYNC)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
