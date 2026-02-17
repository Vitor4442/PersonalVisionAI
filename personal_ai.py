import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class personalAI:
    def __init__(self, file_name = "rosca2cor.mp4"):
        self.file_name = file_name
        self.model_path = "pose_landmarker_full.task"
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
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

    def process_image(self, drawm, display):
        with python.vision.PoseLandmarker.create_from_options(self.options) as landmark:
            cap = cv2.VideoCapture(self.file_name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            calc_ts = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if ret == True:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    calc_ts = int(calc_ts + 1000 / fps)
                    detection_result = landmark.detect_for_video(mp_image, calc_ts)

                    if drawm:
                        frame = self.draw_landmarks_on_image(frame, detection_result)

                    if display:
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break

                else:
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    PersonalAI = personalAI()
    PersonalAI.process_image(drawm=False, display=True)









