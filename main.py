# gesture_recognizer_multi.py
"""
Realtime Multi-Hand Gesture Recognizer
Gestures: Open Palm, Fist, Peace (V-sign), Thumbs Up
Supports up to 2 hands
Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]
MCP_IDS = [2, 5, 9, 13, 17]
WRIST = 0
MIDDLE_MCP = 9

SMOOTH_BUF = 5

def angle3D(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(cosang, -1, 1)
    return np.degrees(np.arccos(cosang))

def finger_states(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm])
    states = [False]*5
    scale = np.linalg.norm(pts[WRIST][:2] - pts[MIDDLE_MCP][:2]) + 1e-6

    for i in range(1,5):
        ang = angle3D(pts[MCP_IDS[i]], pts[PIP_IDS[i]], pts[TIP_IDS[i]])
        states[i] = ang > 165

    ang_thumb = angle3D(pts[MCP_IDS[0]], pts[PIP_IDS[0]], pts[TIP_IDS[0]])
    dist_thumb_index = np.linalg.norm(pts[TIP_IDS[0]][:2] - pts[MCP_IDS[1]][:2]) / scale
    states[0] = ang_thumb > 160 or dist_thumb_index > 0.5

    return states

def classify(states, lm):
    thumb, idx, mid, ring, pinky = states
    n_ext = sum(states)

    if n_ext >= 4:
        return "Open Palm"
    if n_ext == 0:
        return "Fist"
    if idx and mid and not ring and not pinky:
        tip_idx = np.array([lm[TIP_IDS[1]].x, lm[TIP_IDS[1]].y])
        tip_mid = np.array([lm[TIP_IDS[2]].x, lm[TIP_IDS[2]].y])
        if np.linalg.norm(tip_idx - tip_mid) > 0.08:
            return "Peace"
    if thumb and not idx and not mid and not ring and not pinky:
        wrist_y = lm[WRIST].y
        thumb_tip_y = lm[TIP_IDS[0]].y
        if thumb_tip_y < wrist_y - 0.05:
            return "Thumbs Up"
        return "Thumbs Up"
    return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not found")
        return

    preds = deque(maxlen=SMOOTH_BUF)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,              # 👈 allow two hands
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as hands:

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            H,W,_ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gestures = []

            if results.multi_hand_landmarks:
                # compute area to know largest hand
                hand_entries=[]
                for idx,hand in enumerate(results.multi_hand_landmarks):
                    xs=[lm.x for lm in hand.landmark]; ys=[lm.y for lm in hand.landmark]
                    area=(max(xs)-min(xs))*(max(ys)-min(ys))
                    hand_entries.append((area,idx,min(xs),min(ys),max(xs),max(ys)))

                hand_entries.sort(reverse=True,key=lambda x:x[0])

                for area,idx,min_x,min_y,max_x,max_y in hand_entries:
                    hand=results.multi_hand_landmarks[idx]
                    mp_drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    states=finger_states(hand.landmark)
                    gesture=classify(states,hand.landmark)
                    gestures.append((gesture,area))

                    x1,y1=int(min_x*W)-10,int(min_y*H)-10
                    x2,y2=int(max_x*W)+10,int(max_y*H)+10
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),2)
                    cv2.putText(frame,gesture,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

            # choose main gesture from largest hand
            if gestures:
                main_gesture=max(gestures,key=lambda g:g[1])[0]
                preds.append(main_gesture)
            else:
                preds.append("None")

            non_none=[p for p in preds if p!="None"]
            if non_none:
                common=Counter(non_none).most_common(1)[0][0]
            else:
                common="-"

            cv2.putText(frame,f"Main Gesture: {common}",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

            cv2.imshow("Multi-Hand Gesture Recognizer",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
