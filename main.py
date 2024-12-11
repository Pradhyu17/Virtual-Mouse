import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
mouse = Controller() #using controller to control mouse


screen_width, screen_height = pyautogui.size() #we will get l & w by autogui

mpHands = mp.solutions.hands #
hands = mpHands.Hands(
    static_image_mode=False, #coz capturing video not pics
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1 #for making v mouse we just need 1 hand
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None #coz if we get no output let the code continue


def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )

def detect_gesture(frame, landmark_list, processed): #to detect gesture
    if len(landmark_list) >= 21: 

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50  and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist): #if the gestures are correct
            mouse.press(Button.left) #clicking
            mouse.release(Button.left) #releasing
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def main(): #Main fun & Capturing code
    draw = mp.solutions.drawing_utils #draw the landmarks
    cap = cv2.VideoCapture(0) #to capture

    try:
        while cap.isOpened(): #if capture is running successfully
            ret, frame = cap.read() #return the video frame by frame
            if not ret:
                break
            frame = cv2.flip(frame, 1) #flip the frame to look like mirror
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #frame converted to RGB
            processed = hands.process(frameRGB) #Stored in this variable

            landmark_list = [] #Array to recieve landmarks from frame 
            if processed.multi_hand_landmarks: 
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS) #drawing the landmarks
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y)) #pushing landmarks into landmarks list

            detect_gesture(frame, landmark_list, processed) #Calling to detect the gesture

            cv2.imshow('Frame', frame) #Show the frame
            if cv2.waitKey(1) & 0xFF == ord('q'): #q to exit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()#close after done


if __name__ == '__main__':
    main() #this will not run if imported





