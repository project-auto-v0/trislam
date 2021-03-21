import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl
import plotly.express as px
import plotly.graph_objects as go

class FeaturesAndMatching:
    def __init__(self, vidlink):
        self.vidlink = vidlink
        self.vid_over = False #signals if video over
        self.kp_set1 = [] #contains kps for frame1 of each frame pair
        self.kp_set2 = [] #contains kps for frame2 of each frame pair
        self.match_list = [] #contains sets of matches for each frame pair
        
        #get no. of frames in vid
        cap = cv2.VideoCapture(self.vidlink)
        self.vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
    #skip ahead frames no. of frames
    def frames_ahead(self, cap, frames):
        if frames <= 0:
            return 0
        for i in range(frames):
            cap.read()
    
    #get specified frames from vid
    def get_frames(self, frames):
        cap = cv2.VideoCapture(self.vidlink)
        self.frames_ahead(cap, frames[0] - 1)
        ret1, frame1 = cap.read()
        self.frames_ahead(cap, (frames[1] - frames[0]) - 1)
        ret2, frame2 = cap.read()
        cap.release()
        
        self.frame1 = frame1
        self.frame2 = frame2
    
        return self.frame1, self.frame2
    
    #perform orb on specified frames from specified vid, return keypoints, descriptions for both frames and frames themselves
    def perform_orb(self, skip=1, init=False, start_frame=0, draw=False, draw_type="plt"):
        orb = cv2.ORB_create()
        
        if init:
            self.vid_over = False
            self.curr_frame_index = start_frame
            self.get_frames(frames=(self.curr_frame_index, self.curr_frame_index + skip))
            kp1, des1 = orb.detectAndCompute(self.frame1, None)
            kp2, des2 = orb.detectAndCompute(self.frame2, None)
        else:
            self.curr_frame_index += skip       
            self.get_frames(frames=(self.curr_frame_index, self.curr_frame_index + skip))
            kp1, des1 = self.kp2, self.des2 #no need to compute again, previous 2nd is this iteration's first
            kp2, des2 = orb.detectAndCompute(self.frame2, None)
        
        self.kp1 = kp1
        self.des1 = des1
        self.kp2 = kp2
        self.des2 = des2
    
        if draw:
            if draw_type == "plt":
                frame1_kp = cv2.drawKeypoints(self.frame1, kp1, None, color=(0, 255, 0), flags=0)
                frame2_kp = cv2.drawKeypoints(self.frame2, kp2, None, color=(0, 255, 0), flags=0)
        
                plt.subplot(1, 2, 1)
                plt.imshow(frame1_kp)
                plt.subplot(1, 2, 2)
                plt.imshow(frame2_kp)
                plt.show()
        
            elif draw_type == "cv2":
                frame1_kp = cv2.drawKeypoints(self.frame1, kp1, None, color=(0, 255, 0), flags=0)
                cv2.imshow("frame1", frame1_kp)
                cv2.waitKey(1)
        
        #if video over
        if self.curr_frame_index + 2 * skip >= self.vid_len:
            self.vid_over=True  
                
        return (self.kp1, self.des1, self.frame1), (self.kp2, self.des2, self.frame2)
    
    #filter matches by using Lowe's ratio test (knnMatch finds k=2 matches for each point, we only keeps the points whose matches are sufficiently distant)
    def match_filter(self, matches, data1, data2, max_matches_per_pair):
        good_matches = []
        matches_under_consideration = []
        for match1, match2 in matches:
            if match1.distance < 0.7*match2.distance:
                matches_under_consideration.append(match1)
        
        matches_under_consideration = sorted(matches_under_consideration, key=lambda x : x.distance)
        matches_under_consideration = matches_under_consideration[:max_matches_per_pair]
        
        for match in matches_under_consideration:
            kp1 = data1[0]
            kp2 = data2[0]
            p1 = kp1[match.queryIdx].pt
            p2 = kp2[match.trainIdx].pt
            good_matches.append([p1, p2])
        good_matches = np.array(good_matches)
        return good_matches
    
    #perform orb sequentially on the entire video
    def orb_loop(self, skip=1, draw=False):
        data1, data2 = self.perform_orb(skip=skip, init=True, start_frame=0, draw=draw, draw_type="cv2")
        self.kp_set1.append(data1)
        self.kp_set2.append(data2)
        
        i = 0
        while not self.vid_over:
            data1, data2 = self.perform_orb(skip=skip, draw=draw, draw_type="cv2")
            self.kp_set1.append(data1)
            self.kp_set2.append(data2)
            print("orb: " + str(i))
            i += 1
        
    #perform brute force knn matching on kp/des pairs obtained from orb_loop
    def match_loop(self, max_matches_per_pair=40):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.match_list=[]
        i = 0
        for data1, data2 in zip(self.kp_set1, self.kp_set2):
            des1 = data1[1]
            des2 = data2[1]
            matches = matcher.knnMatch(des1, des2, k=2)
            good_matches = self.match_filter(matches, data1, data2, max_matches_per_pair=max_matches_per_pair)
            self.match_list.append(good_matches)
            print("match: " + str(i))
            i += 1
            
        return self.match_list

class StructureFromMotion:
    def __init__(self, match_list):
        self.match_list = match_list
        w, h = 640, 480
        x, y = w/2, h/2
        fov = 60
        self.fov = fov * math.pi / 180
        focus_x = x / math.tan(self.fov / 2)
        focus_y = y / math.tan(self.fov / 2)
        
        #intrinsic matrix
        self.K = np.array([[focus_x, 0, x], [0, focus_y, y], [0, 0, 1]])
        
        #list of camera positions
        self.camera_poses = []
        self.camera_pos = np.array([[0], [0], [0]]) #current camera position
        
        #list of 3d point positions
        self.points = []
        
        self.scale = 5

    def reconstruct_3d(self):
        for matches in self.match_list:
            pts_1 = matches[:, 0]
            pts_2 = matches[:, 1]     
            F, mask = cv2.findFundamentalMat(pts_2, pts_1, cv2.FM_8POINT) #get fundamental matrix
            points, R, t, mask = cv2.recoverPose(F, pts_2, pts_1, self.K, 500)
            R = np.asmatrix(R).I
            E = np.hstack((R, t)) #extrinsic matrix
            for pt in pts_2:
                pt_2d = np.asmatrix([pt[0], pt[1], 1]).T
                P = np.asmatrix(self.K) * np.asmatrix(E)
                pt_3d = np.asmatrix(P).I * pt_2d #2d to 3d point (wrt to frame1 coord system)
                self.points.append([pt_3d[0][0] * self.scale + self.camera_pos[0], pt_3d[1][0] * self.scale + self.camera_pos[1], pt_3d[2][0] * self.scale + self.camera_pos[2]])
            self.camera_poses.append(self.camera_pos)
            self.camera_pos = np.array([self.camera_pos[0] + t[0], self.camera_pos[1] + t[1], self.camera_pos[2] + t[2]])
    
    def points_np_format(self):
        self.camera_poses = np.squeeze(np.array(self.camera_poses))
        self.points = np.squeeze(np.array(self.points))
    
    def build_map_plt(self):
        fig = plt.figure()
        ax = mpl.Axes3D(fig)
        ax.scatter(self.camera_poses[:, 0], self.camera_poses[:, 1], self.camera_poses[:, 2])
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='g')
        
    def build_map_plotly(self):
        fig = go.Figure()
        fig.add_scatter3d(x=self.camera_poses[:, 0], y=self.camera_poses[:, 1], z=self.camera_poses[:, 2], marker=dict(size=4, color="aqua"))
        fig.add_scatter3d(x=self.points[:, 0], y=self.points[:, 1], z=self.points[:, 2], mode="markers", marker=dict(size=3, opacity=0.4))
        fig.show()
        
feat = FeaturesAndMatching("./test1.mp4")
feat.orb_loop(draw=False, skip=2)
match_list = feat.match_loop(max_matches_per_pair=20)

sfm = StructureFromMotion(match_list)
sfm.reconstruct_3d()
sfm.points_np_format()
sfm.build_map_plotly()
