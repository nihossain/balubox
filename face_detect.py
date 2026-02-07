#!/usr/bin/env python3
"""
Simple webcam face detector (Haar cascade)
Press 'q' to quit, 's' to save a snapshot.
"""
import time
import cv2

def main(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check macOS Camera permissions and try a different index (e.g., --camera 1).")
        return

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Haar cascade not found in OpenCV installation")

    fps = 0.0
    last = time.time()
    img_ctr = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last)) if now != last else fps
        last = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Face Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"snapshot_{img_ctr:03d}.png"
            cv2.imwrite(fname, frame)
            print("Saved", fname)
            img_ctr += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()