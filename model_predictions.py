import cv2

face_classifier = cv2.CascadeClassifier('C:/Users/vikas/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainner.yml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)

    if faces is None:
        print("are bhai shakal dekhao")
        break

    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),1)
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))


    try:
        face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        print(result)

        if result[1] < 500:

            confidence = int(100*(1- result[1]/300))

        display_string = str(confidence) + "% confident it is user"

        cv2.putText(frame, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX,1, (255,0,0),1)

        if(confidence > 88):
            cv2.putText(frame, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX,1, (0, 0, 255), 1)
            cv2.imshow("faceRecognizer", frame)

        else:
            cv2.putText(frame, "locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow("faceRecognizer", frame)


    except:
        cv2.putText(frame, "face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))
        cv2.imshow("faceRecognizer", frame)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()