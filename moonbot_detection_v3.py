#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
* @file object2wspose.py
* @author Gustavo Diaz
* @date 30 June 2023
* @brief ROS node interface for publishing yolo detected objects in robot workspace
'''

import numpy as np
import rospy
import tf
import pyrealsense2
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, PoseArray
import time

class JointsDetection(object):
    def __init__(self):

        self.class_dict = {'body':0,'body_connector':1, 'body_connector_occupied':2, 'gripper':3,'leg':4,'leg_connector':5, 'leg_marker':6}

        #camera frames and info
        self.intrinsics = pyrealsense2.intrinsics()
        self.color_frame = []
        self.eye_color_frame = []
        self.depth_frame = []
        self.br = CvBridge()
        self.selected_camera = ["/camera", "/base_camera"]
        self.cam_prefix = self.selected_camera[1]

        self.PUB_YOLO_IMG = True

        self.table_depth = 0

        # Subscribers
        self.camera_info_sub = rospy.Subscriber(self.cam_prefix+'/color/camera_info', CameraInfo, self.getCameraInfo)
        self.img_sub = rospy.Subscriber(self.cam_prefix+'/color/image_raw', Image, self.getColorFrame)
        self.eye_img_sub = rospy.Subscriber(self.cam_prefix+'/color/image_raw', Image, self.getEyeColorFrame)
        self.depth_sub = rospy.Subscriber(self.cam_prefix+'/aligned_depth_to_color/image_raw', Image, self.getDepthFrame)

        # publishers
        self.target_pub = rospy.Publisher('/vs_target_pose', Float64MultiArray, queue_size=10)
        self.target_msg = Float64MultiArray()
        self.img_pub = rospy.Publisher(self.cam_prefix+'/color/yolo_result', Image, queue_size=10)
        self.eye_img_pub = rospy.Publisher(self.cam_prefix+'/color/yolo_result', Image, queue_size=10)
        self.debug_hds_pub = rospy.Publisher('/debug_hds_poses', PoseArray, queue_size=10)

        # Load the pretrained model for object detection
        self.model = YOLO("/home/moonshot/moonshot_ws/src/xarm_object_detection/datasets/moonbot_v3.v4i.yolov8/runs/detect/moonbot_v3.v4i.yolov8.train/weights/best.pt")
        self.display_result = False

        #broadcaster
        self.tf_pose_br = tf.TransformBroadcaster()

        # listener
        self.tf_pose_ls = tf.TransformListener()

        self.body_postions = []
        self.legs_postions = []
        self.bconnector_postions = []
        self.lconnector_positions = []
        self.gripper_postions = []

    # ******************* callback functions *******************
    
    # callback function to store the parameters of the real sense camera
    def getCameraInfo(self, cameraInfo):
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        self.intrinsics.model  = pyrealsense2.distortion.none  
        self.intrinsics.coeffs = [i for i in cameraInfo.D] 

    # callback function to store the color frame from the real sense camera
    def getColorFrame(self, msg):
        bridge = CvBridge()
        try:
            self.color_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def getEyeColorFrame(self, msg):
        bridge = CvBridge()
        try:
            self.eye_color_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    # callback function to store the depth frame from the real sense camera
    def getDepthFrame(self, depth_frame):
        bridge = CvBridge()
        try:
            self.depth_frame = bridge.imgmsg_to_cv2(depth_frame, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    # ***************************** yolo object detection *****************************
    def yolo_detection(self, model, frame, class_id = None, cam = "base", conf=0.75):
        if(class_id is None):
            results = model(source=frame, show=self.display_result, conf=conf, show_labels = False, save=False, verbose = False)[0] # yolo function to predict on a frame using the load model
        else:
            results = model(source=frame, show=self.display_result, conf=conf, show_labels = False, save=False, classes = class_id, verbose = False)[0]
        if len(results)!= 0 :
            xyxy = results.boxes.xyxy # left top corner (x1,y1) and right bottom corner (x2,y2)
            xyxy = xyxy.cpu().numpy()
            xywh = results.boxes.xywh  # center (x,y) and the width and the height
            xywh = xywh.cpu().numpy()
            classes = results.boxes.cls
            classes = classes.cpu().numpy()

            rects = []
            for i in range(len(xywh)):
                obj = [[xywh[i][0],xywh[i][1]],[xywh[i][2],xywh[i][3]],[xyxy[i][0],xyxy[i][1]],[xyxy[i][2],xyxy[i][3]], classes[i]] 
                # [center coordinate] , [width and height bounding box], [top left corner coord], [bottom right corner coord], classe of the detected obj
                rects.append(obj)
            # if self.PUB_YOLO_IMG:
            #     mask_img = results.plot()
            #     self.img_pub.publish(self.br.cv2_to_imgmsg(mask_img))
                # for i in range(len(results)):
                #     if cam == "base":
                #         self.img_pub.publish(self.br.cv2_to_imgmsg(results[i].plot()))
                #     else:
                #         self.eye_img_pub.publish(self.br.cv2_to_imgmsg(results[i].plot()))
            return rects
        else :
            return []

    def pub_yolo_frame_all(self, model, frame, conf=0.75):
        results = model.predict(task = "detect", source=frame, classes=[0,1,4,5,6], conf=conf, show_labels = False, show_conf=True, save=False, verbose = False)[0] # yolo function to predict on a frame using the load model
        if len(results)!= 0 :
            mask_img = results.plot(labels = False)
            self.img_pub.publish(self.br.cv2_to_imgmsg(mask_img))

    def pub_yolo_frame_all_aux(self):
        if (len(self.color_frame) > 0) and (len(self.depth_frame) > 0):
            self.pub_yolo_frame_all(model=self.model, frame=self.color_frame, conf=0.73)

    def yolo_detection_all(self, model, frame):
        results = model.predict(task = "detect", source=frame, classes=[0,1,3,4,5,6], conf=0.7, show_labels = False, show_conf=True, save=False, verbose = False)[0] # yolo function to predict on a frame using the load model
        if len(results)!= 0 :
            xyxy = results.boxes.xyxy # left top corner (x1,y1) and right bottom corner (x2,y2)
            xyxy = xyxy.cpu().numpy()
            xywh = results.boxes.xywh  # center (x,y) and the width and the height
            xywh = xywh.cpu().numpy()
            classes = results.boxes.cls
            classes = classes.cpu().numpy()

            rects = []
            for i in range(len(xywh)):
                obj = [[xywh[i][0],xywh[i][1]],[xywh[i][2],xywh[i][3]],[xyxy[i][0],xyxy[i][1]],[xyxy[i][2],xyxy[i][3]], classes[i]] 
                # [center coordinate] , [width and height bounding box], [top left corner coord], [bottom right corner coord], classe of the detected obj
                rects.append(obj)
            if self.PUB_YOLO_IMG:
                mask_img = results.plot()
                self.img_pub.publish(self.br.cv2_to_imgmsg(mask_img))
                # for i in range(len(results)):
                #     if cam == "base":
                #         mask_img = results[i].plot()
                #         # for point in results.masks.xy[0]:
                #             # print(type(tuple(point)))
                #             # mask_img = cv2.circle(mask_img, (int(tuple(point)[0]), int(tuple(point)[0])), 2, (255,0,0), thickness=1, lineType=8, shift=0)
                #         self.img_pub.publish(self.br.cv2_to_imgmsg(mask_img))
                #     else:
                #         self.eye_img_pub.publish(self.br.cv2_to_imgmsg(results[i].plot()))
            return rects
        else :
            return []

    # def pub_results_yolo(self):
    #     if self.PUB_YOLO_IMG:
    #         for i in range(len(self.pub_results)):
    #             self.img_pub.publish(self.br.cv2_to_imgmsg(self.pub_results[i].plot()))

    # ***************************** object detection handlers *****************************
    def getTableDepth(self, depth_frame):
        x = int(len(depth_frame)/2) #why?
        y = int(len(depth_frame[0])/2)
        table_depth = depth_frame[x,y]
        # transformation from the pixel coordinates to the world coordinates in mm
        table_point_cam_link = pyrealsense2.rs2_deproject_pixel_to_point(self.intrinsics, [x,y], table_depth)
        table_point_base_link = self.getTfBaseLink(table_point_cam_link, "/table_pose_from_base")
        return table_point_base_link[2]

    def reset_positions(self):
        self.body_postions = []
        self.legs_postions = []
        self.bconnector_postions = []
        self.lconnector_positions = []
        self.gripper_postions = []

    # ***************************** corrdinate transformation handlers *****************************
    def gen_pose_type(self, point):
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1
        return pose

    def gen_pose_array(self, points):
        pose_array = PoseArray()
        pose_array.header.frame_id = "link_base"
        for i in range(len(points)):
            pose_array.poses.append(self.gen_pose_type(points[i]))
        pose_array.header.stamp = rospy.Time.now()
        return pose_array

    def pub_hd_poses(self, points):
        pose_array = self.gen_pose_array(points)
        self.debug_hds_pub.publish(pose_array)

    def addTf(self, t, q, frame_name, frame_ref):
        self.tf_pose_br.sendTransform((t[0], t[1], t[2]),
                         q,
                         rospy.Time.now(),
                         frame_name,
                         frame_ref)
    def getTfBaseLink(self, t, frame_name):
        while(not(rospy.is_shutdown())):
            self.addTf((t[0]/1000.0, t[1]/1000.0, t[2]/1000.0), np.array([0,0,0,1]), frame_name, self.cam_prefix+"_color_optical_frame")
            try:
                (trans,rot) = self.tf_pose_ls.lookupTransform('/link_base', frame_name, rospy.Time(0))
                trans[0] = trans[0] * 1000.0 #+ 30
                trans[1] = trans[1] * 1000.0 #+ 15
                trans[2] = trans[2] * 1000.0 #+ 0
                return trans
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # print("error lookup transform")
                continue

    def getPosInWorkspace(self, depth_frame, box, obj_name):
        box_center_xy = box[0]
        box_dim_xy = box[1]
        # target_point = [box_center_xy[0], box_center_xy[1]+box_dim_xy[1]/2] #box's botom center
        target_point = [box_center_xy[0], box_center_xy[1]] #box's center
        xyz_cam_link = self.pixelToPoint(depth_frame, box_center_xy, target_point)
        xyz_base_link = self.getTfBaseLink(xyz_cam_link, obj_name)
        return xyz_base_link

    def pixelToPoint(self, depth_frame, pixel_coord, target_coord):
        # get the depth 
        depth = depth_frame[int(pixel_coord[1])][int(pixel_coord[0])]
        # transformation from the pixel coordinates to the world coordinates in mm
        return pyrealsense2.rs2_deproject_pixel_to_point(self.intrinsics, target_coord, depth)

    def arctan2(self, s, c):
        angle = np.arctan2(s, c)
        if angle>=0:
            return angle
        else:
            return 2 * np.pi + angle

    def clockwise_around_center(self, point):
        diff = point - self.center
        rcos = np.dot(diff, self.center)
        rsin = np.dot([0,0,1], np.cross(diff, self.center))
        return self.arctan2(rsin, rcos)
    
    def sort_points_1(self, points):
        self.center = np.mean(points, axis=0)
        sorted_points = sorted(points, key=self.clockwise_around_center)
        return sorted_points

    def sort_points_2(self, points):
        sorted_points = sorted(points, key=lambda x:x[2])
        return sorted_points

    def distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

    def sort_points_3(self, points, gripper_pose):
        points_ = []
        for i in range(len(points)):
            points_.append([points[i][0], points[i][1], points[i][2], self.distance(points[i], gripper_pose)])
        sorted_points = sorted(points_, key=lambda x:x[3])
        return [sublist[:-1] for sublist in sorted_points]


    # ***************************** Main Program *****************************
    def updateyolo_all(self):
        if (len(self.color_frame) > 0) and (len(self.depth_frame) > 0):
            self.yolo_detection_all(model=self.model, frame=self.color_frame)

    def updateyolo(self):
        if (len(self.color_frame) > 0) and (len(self.depth_frame) > 0):
            # self.table_depth = self.getTableDepth(self.depth_frame)
            # store the associated color frame and depth frame used when detecting objetc
            robot_bodies = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['body'])
            robot_legs = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['leg'])
            body_connector = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['body_connector'])
            leg_connector = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['leg_connector'])

    def updateyoloEye(self):
        if (len(self.eye_color_frame) > 0):
            # self.table_depth = self.getTableDepth(self.depth_frame)
            # store the associated color frame and depth frame used when detecting objetc
            robot_bodies = self.yolo_detection(model=self.model, frame=self.eye_color_frame, class_id=self.class_dict['body'], cam="eye")
            robot_legs = self.yolo_detection(model=self.model, frame=self.eye_color_frame, class_id=self.class_dict['leg'], cam="eye")
            body_connector = self.yolo_detection(model=self.model, frame=self.eye_color_frame, class_id=self.class_dict['body_connector'], cam="eye")
            leg_connector = self.yolo_detection(model=self.model, frame=self.eye_color_frame, class_id=self.class_dict['leg_connector'], cam="eye")

    def updateObjectsPose(self):
            self.reset_positions()
            start = time.time()
            elapse = 0
            while(not(rospy.is_shutdown()) and elapse<0.2):
                elapse = time.time() - start
            if (len(self.color_frame) > 0) and (len(self.depth_frame) > 0):
                robot_legs = []
                # self.table_depth = self.getTableDepth(self.depth_frame)
                # store the associated color frame and depth frame used when detecting objetc
                conf = 0.73
                robot_bodies = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['body'], conf=conf)
                robot_legs = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['leg'], conf=conf)
                body_connector = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['body_connector'], conf=conf)
                leg_connector = self.yolo_detection(model=self.model, frame=self.color_frame, class_id=self.class_dict['leg_connector'], conf=conf)

                self.pub_yolo_frame_all(model=self.model, frame=self.color_frame, conf=conf)

                # self.reset_positions()
                legs_postions = []
                lconnector_positions = []
                bconnector_postions = []
                body_postions = []

                if len(robot_bodies)!=0:
                    # self.body_postions = []
                    for i in range(len(robot_bodies)):
                        body_pose = self.getPosInWorkspace(self.depth_frame, robot_bodies[i], "/robot_body_"+str(i))
                        body_postions.append(body_pose)
                    self.target_msg.data = body_postions[0]
                    # self.target_pub.publish(self.target_msg)
                if len(robot_legs)!=0:
                    legs_postions = []
                    for i in range(len(robot_legs)):
                        leg_pose = self.getPosInWorkspace(self.depth_frame, robot_legs[i], "/leg_"+str(i))
                        legs_postions.append(leg_pose)
                if len(body_connector)!=0:
                    bconnector_postions = []
                    for i in range(len(body_connector)):
                        bconnector_pose = self.getPosInWorkspace(self.depth_frame, body_connector[i], "/body_connector_"+str(i))
                        bconnector_postions.append(bconnector_pose)
                if len(leg_connector)!=0:
                    lconnector_positions = []
                    for i in range(len(leg_connector)):
                        lconnector_pose = self.getPosInWorkspace(self.depth_frame, leg_connector[i], "/leg_connector_"+str(i))
                        lconnector_positions.append(lconnector_pose)

                self.color_frame = []
                self.depth_frame = []
                print("legs_postions_yolo_if: ", legs_postions)
                return [legs_postions, lconnector_positions, bconnector_postions, body_postions]
            self.color_frame = []
            self.depth_frame = []
            return None
            
if __name__ == '__main__':

    rospy.init_node('joints_detection_node', anonymous=False)

    object_detection = JointsDetection()
    while(not(rospy.is_shutdown())):
        # object_detection.updateObjectsPose()
        # object_detection.updateyolo()
        # object_detection.updateyoloEye()
        # object_detection.updateyolo_all()
        object_detection.pub_yolo_frame_all_aux()
