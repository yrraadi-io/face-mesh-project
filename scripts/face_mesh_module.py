import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture(0)
# p_time = 0

# mp_draw = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=2,
#     refine_landmarks=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )
# draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2)

# while True:
#     success, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(img_RGB)
#     # print(results, type(results))

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             mp_draw.draw_landmarks(
#                 frame,
#                 face_landmarks,
#                 mp_face_mesh.FACEMESH_CONTOURS,
#                 draw_spec,
#                 draw_spec,
#             )
#             for id, lm in enumerate(face_landmarks.landmark):
#                 ih, iw, ic = frame.shape
#                 x, y = int(lm.x * iw), int(lm.y * ih)
#                 print(id, x, y)


class FaceMeshDectector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=2, circle_radius=2)

    def find_face_mesh(self, frame, draw=True):
        self.img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.img_RGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        self.draw_spec,
                        self.draw_spec,
                    )
                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(
                    #     frame,
                    #     str(id),
                    #     (x, y),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.1,
                    #     (255, 0, 0),
                    #     1,
                    # )
                    face.append([x, y])
                    # print(id, x, y)
                    faces.append(face)

            return frame, faces


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        detector = FaceMeshDectector()
        frame, faces = detector.find_face_mesh(frame, draw=True)

        if len(faces) > 0:
            print(len(faces))

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

        cv2.imshow("frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
