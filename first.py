from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    return round(angle, 2)  # Rounded for better readability

@app.route('/detect', methods=['POST'])
def detect():
    # Check if file is in request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    
    # Decode image and ensure it's valid
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape  # Get image dimensions

    # Process with MediaPipe Pose
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Convert normalized coordinates to pixel values
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Release memory
        del image, np_img  

        return jsonify({"angle": angle})

    return jsonify({"error": "No pose detected"}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
