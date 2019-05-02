import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
from hand import Hand
from body import Body
from pose_norm import PoseNormalizer
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

source = 'video/neev.mp4'
source_video = cv2.VideoCapture(source)

target = 'video/tompkin.mp4'
target_video = cv2.VideoCapture(target)

# Initialize arrays to store pose information
source_poses = []
target_poses = []

# Initialize array to store subset information
source_subsets = []

# Initialize arrays for PoseNormalizer
source_left = []
source_right = []
target_left = []
target_right = []

# Frame counter
frame = 0

while(True): 
     # reading from frame 
    ret_source, source_frame = source_video.read() 
    ret_target, target_frame = target_video.read()

    sh1 = source_frame.shape
    sh2 = target_frame.shape
        
    if ret_source and ret_target and frame < 200:
        # Resize frames to make computation faster
        # source_frame = cv2.resize(source_frame, (720, 480))
        # target_frame = cv2.resize(target_frame, (720, 480))

        source_frame = np.rot90(np.rot90(np.rot90(source_frame)))
        h, w, layers = source_frame.shape
        source_frame = cv2.resize(source_frame, (int(w/2), int(h/2)))

        target_frame = np.rot90(np.rot90(np.rot90(target_frame)))
        target_frame = cv2.resize(target_frame, (int(w/2), int(h/2)))

        # Grab pose estimations for both video frames
        source_candidate, source_subset = body_estimation(source_frame)
        target_candidate, target_subset = body_estimation(target_frame)

        # Put target subset into memory
        source_subsets.append(source_subset)

        # Put pose estimations into memory
        source_poses.append(source_candidate)
        target_poses.append(target_candidate)

        # Grab ankles
        source_left.append(source_candidate[13, 1])
        source_right.append(source_candidate[10, 1])
        target_left.append(target_candidate[13, 1])
        target_right.append(target_candidate[10, 1])

        frame += 1
        print(frame)
    else:
        break

source_video.release()  
target_video.release()


source_dict = {
  "left": np.array(source_left),
  "right": np.array(source_right)
}

target_dict = {
  "left": np.array(target_left),
  "right": np.array(target_right)
}

pose_normalizer = PoseNormalizer(source_dict, target_dict, epsilon=3)

# Testing video output
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video=cv2.VideoWriter('video/pose_normed.mp4', fourcc, 30,(540, 960))
frames = len(source_poses)
transformed_all = pose_normalizer.transform_pose_global(source_poses, target_poses)
for i in range(frames):
    pose = transformed_all[i]
    # Visually testing if normalizing works 
    canvas = np.ones((960, 540, 3), dtype='uint8') * 255
    canvas = util.draw_bodypose(canvas, pose, source_subsets[i])
    video.write(canvas)
     
print(pose_normalizer.statistics)
