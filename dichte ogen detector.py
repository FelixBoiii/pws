import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import winsound

# open haar-like classifier
face_cascade = cv2.CascadeClassifier(
    r"E:\Users\Bart\Documents\GitHub\pws\haarcascade_frontalface_default.xml"
)
# laad keras model in
model_1 = keras.models.load_model("model_1")
# camera setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# overige variablen
cropped_image = 0
resized = 0
avPredictionStep = 0
avPredictionPerc = 0
avSize = 10
avPrediction = np.full(avSize, 50)
rectColor = (0, 255, 255)

while 1:
    # leest de webcam af
    ret, img = cap.read()
    # maakt grayscale image voor haar-like en neurale netwerk
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # doet de haar-like detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # als er gezichten zijn word er door alle gezichten gelopen en word het meest rechter gezicht opgeslagen
    # omdat dat het gezicht is van de bestuurder
    if len(faces) != 0:
        x, y, w, h = 0, 0, 5, 5
        for (xn, yn, wn, hn) in faces:
            if xn > x:
                x = xn
                y = yn
                w = wn
                h = hn

        # maakt het vierkant om het gezicht en creeerd de afbeelding voor het neurale netwerk
        cv2.rectangle(img, (x - 5, y - 5),
                      (x + w + 5, y + h + 5), rectColor, 3)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        cropped_image = gray[y: y + h, x: x + w]
        resized = cv2.resize(cropped_image, [100, 100])

    # verander array vorm om hetzelfde te zijn als de input van het neurale netwerk: 100x100 --> 1x100x100x1
    test_image = (resized[..., ::-1].astype(np.float32)) / 255.0
    img_array = np.expand_dims(test_image, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # neurale netwerk doet hier de voorspelling. de andere twee zinnen zijn voor het omzetten naar integer
    # en de gemmidelde voorspelling te nemen van de laatste {avSize} aantal voorspellingen
    prediction = int(float(str(model_1.predict(img_array))[2:-2])*100)
    avPrediction[avPredictionStep] = prediction
    avPredictionPerc = int(np.average(avPrediction))

    # voegt de tekst toe aan het webcam beeld
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, str(avPredictionPerc)+"%", (10, 80), font,
                3, rectColor, 3, cv2.LINE_AA)

    # maakt het biep geluid als de gemmidelde voorspelling onder 10% valt
    if avPredictionPerc < 10:
        rectColor = (0, 0, 255)
        winsound.Beep(1000, 100)
    else:
        rectColor = (0, 255, 255)

    # zorg ervoor dat de voorspellings array goed update
    if avPredictionStep < avSize-1:
        avPredictionStep += 1
    else:
        avPredictionStep = 0

    # maakt de twee windows aan voor het webcam beeld en het grayscale hoofd. De tweede is niet persee nodig
    # en is meer bedoelt voor duidelijkheid
    cv2.imshow("img", img)
    cv2.imshow("img2", resized)

    # verbreekt de while loop als er op 'q' wordt gedrukt
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# als de while loop verbroken wordt door de 'q' sluit dit het programma
cap.release()
cv2.destroyAllWindows()
