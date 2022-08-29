import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_RGB)
    # print(results, type(results))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                draw_spec,
                draw_spec,
            )
            for id, lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

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
