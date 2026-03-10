import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
from gpiozero import Buzzer

# ---------- BUZZER ----------
buzzer = Buzzer(17)   # GPIO17 (Pin 11)

# ---------- FUNCTION ----------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ---------- PARAMETERS ----------
thresh = 0.25
frame_check = 20

# ---------- LOAD MODELS ----------
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ---------- CAMERA ----------
cap = cv2.VideoCapture("libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera not detected")
    exit()

flag = 0

# ---------- MAIN LOOP ----------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    for subject in subjects:

        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        # ---------- DROWSINESS DETECTION ----------
        if ear < thresh:

            flag += 1
            print(flag)

            if flag >= frame_check:

                buzzer.on()

                cv2.putText(frame, "DROWSINESS ALERT!",
                            (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

        else:
            flag = 0
            buzzer.off()

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# ---------- CLEANUP ----------
cap.release()
buzzer.off()
cv2.destroyAllWindows()