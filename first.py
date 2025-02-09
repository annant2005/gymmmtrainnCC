from flask import Flask, render_template, Response, jsonify, request  # Added request import
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture (camera)
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    return angle

def gen_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the BGR frame to RGB for pose detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose detection
        results = pose.process(frame_rgb)

        # Draw landmarks if detected
        if results.pose_landmarks:
            # Get the landmarks
            landmarks = results.pose_landmarks.landmark

            # Define shoulder, elbow, and wrist for angle calculation
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle between shoulder, elbow, and wrist
            angle = calculate_angle(shoulder, elbow, wrist)

            # Display the angle on the screen
            cv2.putText(frame, f'Angle: {int(angle)}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the frame to JPEG for web display
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the correct format for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    # Render the main page with the video feed
    return render_template('index.html')

@app.route('/video')
def video():
    # Stream the video feed with pose detection
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the image from the POST request
    file = request.files['image']
    file.save("temp.jpg")

    image = cv2.imread("temp.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            return jsonify({"angle": angle})

    return jsonify({"error": "No pose detected"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
