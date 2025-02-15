import cv2 
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
option_index = 0
options = ["Option 1", "Option 2", "Option 3", "Option 4"]
previous_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color (BGR to RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            index_y = int(index_tip.y * frame_height)
            wrist_y = int(wrist.y * frame_height)

            # Detect upward/downward movement
            if previous_y is not None:
                if wrist_y < previous_y - 20:  # Move hand up
                    option_index = (option_index + 1) % len(options)
                elif wrist_y > previous_y + 20:  # Move hand down
                    option_index = (option_index - 1) % len(options)

            previous_y = wrist_y

            # Detect index finger extension (Selection)
            if index_tip.y < index_pip.y - 0.05:
                print(f"Selected: {options[option_index]}")
                pyautogui.press("enter")  # Simulate Enter key

            # Detect two-finger gesture (Go Back)
            if (index_tip.y < index_pip.y - 0.05) and (middle_tip.y < middle_pip.y - 0.05):
                print("Go Back Command")
                pyautogui.press("backspace")

            # Detect fist (Exit App)
            if (index_tip.y > index_pip.y) and (middle_tip.y > middle_pip.y):
                print("Exit Application")
                pyautogui.press("esc")
                break

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display options
    for i, option in enumerate(options):
        color = (0, 255, 0) if i == option_index else (255, 255, 255)
        cv2.putText(frame, option, (50, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
