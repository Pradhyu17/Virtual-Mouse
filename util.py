import numpy as np

def get_angle(a, b, c): #get the angle b/w two fingers
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians)) #convert to degree
    return angle


def get_distance(landmark_ist): 
    if len(landmark_ist) < 2: #we need two point to calculate
        return
    (x1, y1), (x2, y2) = landmark_ist[0], landmark_ist[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000]) #Interpolite higher value