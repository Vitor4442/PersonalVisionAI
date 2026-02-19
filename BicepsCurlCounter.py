import numpy as np
import math

class BicepsCurlCounter:
    def __init__(self):
        self.counter = 0
        self.stage = None  # "up" ou "down"

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def update(self, detection_result, image_width, image_height):
        if not detection_result.pose_landmarks:
            return self.counter, None

        landmarks = detection_result.pose_landmarks[0]

        # Landmarks do braço direito (lado)
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]

        shoulder_coords = (shoulder.x * image_width, shoulder.y * image_height)
        elbow_coords = (elbow.x * image_width, elbow.y * image_height)
        wrist_coords = (wrist.x * image_width, wrist.y * image_height)

        angle = self.calculate_angle(shoulder_coords, elbow_coords, wrist_coords)

        # Lógica da repetição
        if angle > 150:
            self.stage = "down"

        if angle < 65 and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        return self.counter, angle
