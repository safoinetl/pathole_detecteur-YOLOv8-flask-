# from flask import Flask, render_template, request
# from ultralytics import YOLO

# app = Flask(__name__ ,static_url_path='/static')
# model = YOLO('pathole_model.pt')

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/detect', methods=['POST'])
# def detect():
#     result = model.predict(source=0, imgsz=640, conf=0.6, show=True)
#     return result.imgs[0]

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__, static_url_path='/static')
model = YOLO('pathole_model.pt')

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to start video feed
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the camera
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run the YOLOv8 model to detect potholes
            results = model.predict(source=frame, imgsz=640, conf=0.4)

            # Draw results on the frame
            annotated_frame = results[0].plot()

            # Encode the frame as a JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/detect')
def detect():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
