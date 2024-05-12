#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:11:07 2023

@author: fmry
"""

#%% Sources

#Source: https://github.com/YomnaAhmed97/Head-Pose-Estimation/blob/main/Head_pose_estimation_.ipynb

#%% Modules

import jax.numpy as jnp
import numpy as np
import cv2,glob
import scipy.io as sio
from math import cos, sin
from pathlib import Path
import mediapipe
import warnings
warnings.filterwarnings('ignore')

import argparse

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--path', default="../../../Data/AFLW/AFLW2000/",
                        type=str)

    args = parser.parse_args()
    return args

#%% Drawing

def draw_axis(img, pitch,yaw,roll, tdx=None, tdy=None, size = 100):

    yaw = -yaw
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

#%% train for (x,y,t)

def load_aflw2000()->None:
    
    args = parse_args()
    
    # X_points, Y_points, labels and detected files which are the images thar MediaPipe was able to detect the face  
    images = []
    x_points= []
    y_points = []
    labels = []
    detected_files = []
    
    # extracting the file names (2000 name)
    file_names = sorted([Path(f).stem for f in glob.glob(''.join((args.path, '*.mat')))])
    
    # detecting faces and extracting the points
    faceModule = mediapipe.solutions.face_mesh
    # looping over the file names to load the images and their corresponding mat file
    for filename in file_names:
        with faceModule.FaceMesh(static_image_mode=True) as faces:
            # loading the image
            image = cv2.imread(''.join((args.path, filename,'.jpg')))
            # processing the image to detect the face and then generating the land marks (468 for each x,y,z).
            results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks != None: 
                # appending the file names where have been detected.
                detected_files.append(filename)
                # detecting the face
                face = results.multi_face_landmarks[0]
                # initializing two lists to store the points for the image.
                X = []
                Y = []
                # looping over the 468 points of x and y
                for landmark in face.landmark:
                    x = landmark.x
                    y = landmark.y
                    # note: the x and y values are scaled to the their width and height so we will get back their actual value in the image.
                    shape = image.shape 
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])
                    # X_features
                    X.append(relative_x)
                    # Y_features
                    Y.append(relative_y)
    
                # converting the lists to numpy arrays
                X = np.array(X)
                Y = np.array(Y)
                # appending the points of the images in the list of all image points
                images.append(image)
                x_points.append(X)
                y_points.append(Y)
    
                # loading the mat file to extract the labels (pitch,yaw,roll)
                mat_file = sio.loadmat(''.join((args.path,filename,'.mat')))
                # extracting the labels 3 angels
                pose_para = mat_file["Pose_Para"][0][:3]
                # appending the 3 angels to labels list
                labels.append(pose_para)
    
    # converting features and labels to 2D array
    x_points = jnp.stack(x_points)
    y_points = jnp.stack(y_points)
    labels = jnp.stack(labels)
    images = jnp.stack(images)
    labels = labels/jnp.linalg.norm(labels, axis=-1).reshape(-1,1)
    
    x_center = x_points - x_points[:,99].reshape(-1,1)
    y_center = y_points - y_points[:,99].reshape(-1,1)
    
    # normalizing the data 
    X_171 = x_points[:,171]
    X_10 = x_points[:,10]
    Y_171 = y_points[:,171]
    Y_10 = y_points[:,10]
    # computing the distance
    distance = np.linalg.norm(np.array((X_10,Y_10)) - np.array((X_171,Y_171)),axis = 0).reshape(-1,1)
    Norm_x = x_center / distance
    Norm_y = y_center / distance
    final_x = Norm_x
    final_y = Norm_y
    features = np.hstack([final_x,final_y])
    
    
    data = {'features': features,
            'xlandmarks': x_points,
            'ylandmarks': y_points,
            'images': images,
            'labels': labels}
    
    
    
    
    return

#%% Main

if __name__ == '__main__':
        
    load_aflw2000()
    
