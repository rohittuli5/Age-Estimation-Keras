import numpy as np
import cv2
import keras
from numpy import asarray
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras.layers as L
from keras import Sequential
import keras
from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import activations

model = tf.keras.Sequential([
    L.InputLayer(input_shape=(64,64,1)),
    L.Conv2D(32, (3, 3), activation='relu'),
    L.BatchNormalization(),
    L.MaxPooling2D((2, 2)),
    L.Conv2D(64, (3, 3), activation='relu'),
    L.MaxPooling2D((2, 2)),
    L.Conv2D(128, (3, 3), activation='relu'),
    L.MaxPooling2D((2, 2)),
    L.Flatten(),
    L.Dense(64, activation='relu'),
    L.Dropout(rate=0.2),
    L.Dense(1, activation='relu')
])

#sgd = tf.keras.optimizers.SGD(momentum=0.9)

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                              patience=5, min_lr=0.001)
video_stream=cv2.VideoCapture(0)
faces=[]
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model.load_weights('abs_age_itself_model.h5')
while(True):
    ret, frame=video_stream.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_image=frame[y-20:y+h+20,x-20:x+w+20]
        arr=[]
        crop_image=cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        crop_image=asarray(crop_image).astype('float32')
        crop_image=cv2.resize(crop_image,(64,64), interpolation=cv2.INTER_AREA)
        
        arr.append(crop_image)
        arr=np.array(arr)
        arr.reshape(-1,64,64,1)
        arr=np.expand_dims(arr, axis=3)
        output=model.predict(arr)
        res=str(int(output[0][0]))
        if len(output)>0 and len(output[0])>0:
            cv2.putText(frame,res,(x,y), cv2.FONT_HERSHEY_SIMPLEX,1 ,(0,0,255),2,cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(33) == ord('a'):
        break

video_stream.release()
cv2.destroyAllWindows()