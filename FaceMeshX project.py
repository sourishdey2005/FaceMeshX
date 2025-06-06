import cv2
import mediapipe as mp
import time
from math import hypot

# Initialization of  MediaPipe Face Mesh and drawing utilitis 
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def euclidean_distance(p1, p2):
    return hypot(p1.x - p2.x, p1.y - p2.y)

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = euclidean_distance(p2, p6)
    B = euclidean_distance(p3, p5)
    C = euclidean_distance(p1, p4)
    ear = (A + B) / (2.0 * C)
    return ear

# Smile detection using mouth corners and lips
def smile_ratio(landmarks):
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    mouth_width = euclidean_distance(left_corner, right_corner)
    mouth_height = euclidean_distance(upper_lip, lower_lip)

    if mouth_height == 0:
        return 0
    ratio = mouth_width / mouth_height
    return ratio

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

EAR_THRESHOLD = 0.27
SMILE_THRESHOLD = 3.5 
HEAD_TILT_THRESHOLD = 0.02

cap = cv2.VideoCapture(0)
prev_time = 0
screenshot_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            #  face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

            # Head Direction Identification 
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]

            mid_eye_x = (left_eye.x + right_eye.x) / 2
            threshold = 0.015
            diff = nose_tip.x - mid_eye_x
            if diff < -threshold:
                head_direction = "Looking Left"
            elif diff > threshold:
                head_direction = "Looking Right"
            else:
                head_direction = "Looking Center"

            cv2.putText(frame, head_direction, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

            # Head Tilt Identification 
            eye_vertical_diff = left_eye.y - right_eye.y
            if eye_vertical_diff > HEAD_TILT_THRESHOLD:
                head_tilt = "Head Tilt Right"
            elif eye_vertical_diff < -HEAD_TILT_THRESHOLD:
                head_tilt = "Head Tilt Left"
            else:
                head_tilt = "Head Straight"

            cv2.putText(frame, head_tilt, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

            # Eye Aspect Ratio  for eye open/closed
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            eye_status = "Eyes Open" if avg_ear >= EAR_THRESHOLD else "Eyes Closed"
            cv2.putText(frame, eye_status, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Smile detection
            ratio = smile_ratio(landmarks)
            if ratio > SMILE_THRESHOLD:
                smile_status = "Smiling "
            else:
                smile_status = "Not Smiling"

            cv2.putText(frame, smile_status, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Algorithms 
    cv2.putText(frame, 'Press S to Save Screenshot | Q to Quit', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)

    cv2.imshow('FaceMeshX - MediaPipe Advanced', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and results.multi_face_landmarks:
        cv2.imwrite(f"screenshot_mediapipe_{screenshot_count}.png", frame)
        print(f"[INFO] Screenshot saved as screenshot_mediapipe_{screenshot_count}.png")
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
