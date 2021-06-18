from flask import Flask, render_template, Response
import cv2 

app = Flask(__name__)

#camera from local webcam (shows lily camera)
cam = cv2.VideoCapture(0)

#creating the camera frames
def create_frames():
    #while camera "on"
    while True:
        #check if the camera was read
        cam_read, frame = cam.read()
        if not cam_read: #if camera not read, stop
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame) #compress frames
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #add each frame together and show result

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/webcam')
def webcam():
    return Response(create_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)