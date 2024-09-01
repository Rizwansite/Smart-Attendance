import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = '/Users/aliab/OneDrive/Desktop/images/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(data_path, onlyfiles[i])
    images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    # Extract student ID from the filename
    student_id = int(file.split('_')[1])  # Extracting the ID from the filename
    Labels.append(student_id)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print('Model Trained')

face_classifier = cv.CascadeClassifier('dataset/hasscode_classifire_frontalFace.xml')

# Check if the classifier is initialized successfully
if face_classifier.empty():
    print('Error loading classifier')
    exit()

print('Classifier loaded successfully')


def face_detect(img,size=0.5):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h,x:x+w]
        roi = cv.resize(roi,(200,200))

    return img, roi

cap = cv.VideoCapture(0)


while True:
    ret, frame = cap.read()
    image, frame = face_detect(frame)

    try:
        face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        res = model.predict(face)

        if res[1] < 500:
            student_id = res[0]  # Recognized student ID
            confidence = int(100 * (1 - (res[1]) / 300))

            if confidence > 82:
                cv.putText(image, str(student_id), (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('Face Cropper', image)
            else:
                cv.putText(image, 'Unknown', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv.imshow('Face Cropper', image)

    except:
        cv.putText(image, 'Face not found', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Face Cropper', image)
        pass

    if cv.waitKey(1) == 13:
        break

cap.release()
cv.destroyAllWindows()
