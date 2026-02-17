import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import cv2
import numpy as np

file_name = "rosca2cor.mp4"
model_path = "pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

with python.vision.PoseLandmarker.create_from_options(options) as landmark:
    cap = cv2.VideoCapture(file_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    calc_ts = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            calc_ts = int(calc_ts + 1000/fps)
            detection_result = landmark.detect_for_video(mp_image, calc_ts)
            annoted_image = draw_landmarks_on_image(frame, detection_result)

            cv2.imshow("Frame", annoted_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
