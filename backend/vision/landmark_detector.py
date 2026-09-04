import os
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker.task")
FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")


class LandmarkDetector:
    def __init__(self):
        self._check_models()

        self.hand_detector = self._create_hand_detector()
        self.pose_detector = self._create_pose_detector()
        self.face_detector = self._create_face_detector()

        self.start_time = time.perf_counter()
        self.frame_count = 0

    def _check_models(self):
        required_models = {
            "Hand": HAND_MODEL,
            "Pose": POSE_MODEL,
            "Face": FACE_MODEL,
        }

        missing_models = []

        for name, path in required_models.items():
            if not os.path.isfile(path):
                missing_models.append(f"{name}: {path}")

        if missing_models:
            message = "\n".join(missing_models)

            raise FileNotFoundError(
                "\nMissing MediaPipe model files:\n\n"
                f"{message}\n\n"
                "Download the required .task files and place them "
                "inside the SignSync/models directory."
            )

    def _create_hand_detector(self):
        base_options = python.BaseOptions(
            model_asset_path=HAND_MODEL
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        return vision.HandLandmarker.create_from_options(options)

    def _create_pose_detector(self):
        base_options = python.BaseOptions(
            model_asset_path=POSE_MODEL
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        return vision.PoseLandmarker.create_from_options(options)

    def _create_face_detector(self):
        base_options = python.BaseOptions(
            model_asset_path=FACE_MODEL
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        return vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        hand_result = self.hand_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        pose_result = self.pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_result = self.face_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return hand_result, pose_result, face_result

    def draw_hand_landmarks(self, frame, hand_result):
        if not hand_result.hand_landmarks:
            return

        height, width = frame.shape[:2]

        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (0, 17),
        ]

        for hand in hand_result.hand_landmarks:
            points = []

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            for start, end in connections:
                if start < len(points) and end < len(points):
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

    def draw_pose_landmarks(self, frame, pose_result):
        if not pose_result.pose_landmarks:
            return

        height, width = frame.shape[:2]

        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
        ]

        for pose in pose_result.pose_landmarks:
            points = []

            for landmark in pose:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (255, 0, 0),
                    -1
                )

            for start, end in connections:
                if start < len(points) and end < len(points):
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 0),
                        2
                    )

    def draw_face_landmarks(self, frame, face_result):
        if not face_result.face_landmarks:
            return

        height, width = frame.shape[:2]

        for face in face_result.face_landmarks:
            for landmark in face:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 0, 255),
                    -1
                )

    def draw_landmarks(
        self,
        frame,
        hand_result,
        pose_result,
        face_result
    ):
        self.draw_hand_landmarks(
            frame,
            hand_result
        )

        self.draw_pose_landmarks(
            frame,
            pose_result
        )

        self.draw_face_landmarks(
            frame,
            face_result
        )

        return frame

    def get_fps(self):
        self.frame_count += 1

        elapsed = time.perf_counter() - self.start_time

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()
        self.face_detector.close()


def main():
    detector = LandmarkDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        detector.close()
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("SignSync landmark detection started.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                time.perf_counter() * 1000
            )

            hand_result, pose_result, face_result = (
                detector.process_frame(
                    frame,
                    timestamp_ms
                )
            )

            frame = detector.draw_landmarks(
                frame,
                hand_result,
                pose_result,
                face_result
            )

            fps = detector.get_fps()

            cv2.putText(
                frame,
                "SignSync - Landmark Detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                "SignSync",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()