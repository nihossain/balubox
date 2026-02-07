import cv2

def get_macbook_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Failed to open webcam")
    return cap

def detect_human_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_rects = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return face_rects

def webcam_face_detection():
    cap = get_macbook_webcam()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_rects = detect_human_faces(frame)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Webcam Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_face_detection()