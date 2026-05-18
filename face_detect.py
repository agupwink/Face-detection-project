import os
import sys
import cv2

CONFIDENCE_THRESHOLD = 0.5
BOX_COLOR = (0, 215, 255)       # gold/yellow box
BADGE_COLOR = (0, 215, 255)     # badge background
TEXT_COLOR = (0, 0, 0)          # black text on badge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(BASE_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


def main():
    for path in (PROTOTXT, CAFFEMODEL):
        if not os.path.exists(path):
            print(f"Error: model file not found: {path}")
            sys.exit(1)

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        print("On macOS, grant Terminal camera access in System Settings > Privacy & Security > Camera.")
        sys.exit(1)

    print("Face detection running — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            label = f"  HUMAN DETECTED  {confidence * 100:.1f}%  "
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y = y1 - th - 10 if y1 - th - 10 > 0 else y2 + 5
            cv2.rectangle(frame, (x1, badge_y), (x1 + tw, badge_y + th + baseline + 4), BADGE_COLOR, cv2.FILLED)
            cv2.putText(frame, label, (x1, badge_y + th + 2), font, font_scale, TEXT_COLOR, thickness)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
