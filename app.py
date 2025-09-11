from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load Haar cascades
face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

tp, fp, fn = 0, 0, 0   # metrics counters


def generate_frames():
    global tp, fp, fn
    cap = cv2.VideoCapture(0)   # open camera INSIDE generator

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detections
        faces_gt = face_cascade1.detectMultiScale(gray, 1.3, 5)
        faces_pred = face_cascade2.detectMultiScale(gray, 1.3, 5)

        # draw
        for (x, y, w, h) in faces_gt:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y, w, h) in faces_pred:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # metrics calc (simplified for demo)
        tp += len(faces_gt)
        fp += max(0, len(faces_pred) - len(faces_gt))

        # encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/metrics')
def metrics():
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0
    recall = tp / (tp+fn) if (tp+fn) > 0 else 0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2)
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

