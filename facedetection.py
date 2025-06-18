import cv2
import numpy as np
from fer import FER


cap = cv2.VideoCapture(0)

detector = FER()

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

 
    emotion_data = detector.detect_emotions(frame)
    
    if emotion_data:
        for emotion in emotion_data:
            (x, y, w, h) = emotion['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
           
            dominant_emotion = emotion['emotions']
            max_emotion = max(dominant_emotion, key=dominant_emotion.get)
            cv2.putText(frame, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()