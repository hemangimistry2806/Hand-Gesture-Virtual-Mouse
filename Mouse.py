import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# --- OPTIMIZED SETTINGS FOR i5 LAPTOP ---
W_CAM, H_CAM = 640, 480
FRAME_R = 100
SMOOTHING = 7
JITTER_THRESHOLD = 5

pyautogui.FAILSAFE = False 
SCR_W, SCR_H = pyautogui.size()

# Variables
pX, pY = 0, 0
cX, cY = 0, 0
pTime = 0

# Quit Variables
fist_detected_time = 0
holding_fist = False
QUIT_HOLD_DURATION = 2   # Hold fist 2 seconds to exit

# --- Initialization ---
cap = cv2.VideoCapture(0)
cap.set(3, W_CAM)
cap.set(4, H_CAM)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# IDs
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_FINGER_TIP = 20
THUMB_TIP = 4
CLICK_THRESHOLD = 40


# ---------------------
# SUPER STABLE FIST DETECTION FUNCTION
# ---------------------
def is_fist(lmList):
    """Returns True if all fingers + thumb are folded."""
    fingers_down = []

    # Fingers except thumb
    for tip in [8, 12, 16, 20]:
        if lmList[tip][2] > lmList[tip - 2][2]:
            fingers_down.append(True)
        else:
            fingers_down.append(False)

    # Thumb fold detection (left/right hand supported)
    thumb_folded = lmList[THUMB_TIP][1] < lmList[THUMB_TIP - 1][1] or \
                   lmList[THUMB_TIP][1] > lmList[THUMB_TIP - 1][1]

    return all(fingers_down) and thumb_folded


# --------------- MAIN LOOP ---------------
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                lmList.append([id, x, y])

    if lmList:
        x_index, y_index = lmList[INDEX_FINGER_TIP][1], lmList[INDEX_FINGER_TIP][2]
        x_middle, y_middle = lmList[MIDDLE_FINGER_TIP][1], lmList[MIDDLE_FINGER_TIP][2]
        x_thumb, y_thumb = lmList[THUMB_TIP][1], lmList[THUMB_TIP][2]

        # Midpoint for cursor
        control_x = (x_index + x_middle) // 2
        control_y = (y_index + y_middle) // 2

        # Draw Control Box
        cv2.rectangle(img, (FRAME_R, FRAME_R), (W_CAM - FRAME_R, H_CAM - FRAME_R),
                      (255, 0, 255), 2)


        # -----------------------------
        # 1. FIST DETECTION (EXIT)
        # -----------------------------
        if is_fist(lmList):
            if not holding_fist:
                fist_detected_time = time.time()
                holding_fist = True

            elapsed = time.time() - fist_detected_time
            countdown = QUIT_HOLD_DURATION - elapsed

            if countdown <= 0:
                break
            else:
                cv2.putText(img, f"Hold Fist to Quit: {countdown:.1f}s",
                            (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)

        else:
            holding_fist = False

            # -----------------------------
            # 2. MOVEMENT LOGIC
            # -----------------------------
            x_map = np.interp(control_x, (FRAME_R, W_CAM - FRAME_R), (0, SCR_W))
            y_map = np.interp(control_y, (FRAME_R, H_CAM - FRAME_R), (0, SCR_H))

            move_distance = math.hypot(x_map - pX, y_map - pY)

            # Jitter filter
            if move_distance > JITTER_THRESHOLD:
                cX = pX + (x_map - pX) / SMOOTHING
                cY = pY + (y_map - pY) / SMOOTHING

                try:
                    pyautogui.moveTo(int(cX), int(cY))
                except:
                    pass

                pX, pY = cX, cY

            cv2.circle(img, (control_x, control_y), 10, (255, 0, 255), cv2.FILLED)


            # -----------------------------
            # 3. CLICK DETECTION
            # -----------------------------
            dist_index_thumb = np.hypot(x_index - x_thumb, y_index - y_thumb)
            dist_middle_thumb = np.hypot(x_middle - x_thumb, y_middle - y_thumb)

            if dist_index_thumb < CLICK_THRESHOLD:
                cv2.circle(img, (x_index, y_index), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click(button="left")
                time.sleep(0.2)

            elif dist_middle_thumb < CLICK_THRESHOLD:
                cv2.circle(img, (x_index, y_index), 15, (0, 0, 255), cv2.FILLED)
                pyautogui.click(button="right")
                time.sleep(0.2)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Virtual Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
