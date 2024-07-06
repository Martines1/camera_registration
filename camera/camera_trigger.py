import json
import os
import sys
import cv2
import numpy as np
import open3d as o3d
from harvesters.core import Harvester
import struct

class Device:
    def __init__(self, device_list):
        self.device_list = device_list
        if sys.platform == "win32":
            cti_file_path_suffix = "/API/bin/photoneo.cti"
        else:
            cti_file_path_suffix = "/API/lib/photoneo.cti"
        self.cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
        self.h = Harvester()
        self.h.add_file(self.cti_file_path, True, True)
        self.h.update()
        self.connected = False
        self.features = []
        self.ia = []
        self.pcd = [None for _ in  range(len(device_list))]
        self.depth = [None for _ in  range(len(device_list))]
        self.image = [None for _ in  range(len(device_list))]

    def display_depth_map_if_available(self, color_component):
        if color_component.width == 0 or color_component.height == 0:
            return
        image = color_component.data.reshape(color_component.height, color_component.width).copy()
        return image

    def display_color_image_if_available(self, color_component):
        if color_component.width == 0 or color_component.height == 0:
            return
        # Reshape 1D array to 2D RGB image
        color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
        # Normalize array to range 0 - 65535
        color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        return color_image

    def display_pointcloud_if_available(self, pointcloud_comp, normal_comp, texture_comp, texture_rgb_comp):
        if pointcloud_comp.width == 0 or pointcloud_comp.height == 0:
            return
        
        # Reshape for Open3D visualization to N x 3 arrays
        pointcloud = pointcloud_comp.data.reshape(pointcloud_comp.height * pointcloud_comp.width, 3).copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)

        if normal_comp.width > 0 and normal_comp.height > 0:
            norm_map = normal_comp.data.reshape(normal_comp.height * normal_comp.width, 3).copy()
            pcd.normals = o3d.utility.Vector3dVector(norm_map)

        # Reshape 1D array to 2D (3 channel) array with image size
        texture_rgb = np.zeros((pointcloud_comp.height * pointcloud_comp.width, 3))
        if texture_comp.width > 0 and texture_comp.height > 0:
            texture = texture_comp.data.reshape(texture_comp.height, texture_comp.width, 1).copy()
            texture_rgb[:, 0] = np.reshape(1/65536 * texture, -1)
            texture_rgb[:, 1] = np.reshape(1/65536 * texture, -1)
            texture_rgb[:, 2] = np.reshape(1/65536 * texture, -1)
        elif texture_rgb_comp.width > 0 and texture_rgb_comp.height > 0:
            texture = texture_rgb_comp.data.reshape(texture_rgb_comp.height, texture_rgb_comp.width, 3).copy()
            texture_rgb[:, 0] = np.reshape(1/65536 * texture[:, :, 0], -1)
            texture_rgb[:, 1] = np.reshape(1/65536 * texture[:, :, 1], -1)
            texture_rgb[:, 2] = np.reshape(1/65536 * texture[:, :, 2], -1)
        else:
            return
        texture_rgb = cv2.normalize(texture_rgb, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        pcd.colors = o3d.utility.Vector3dVector(texture_rgb)
        return pcd

    def connect(self):
        if not self.connected:
            for device_id in self.device_list:
                device_id = "PhotoneoTL_DEV_" + device_id
                print("Connecting device: {}".format(device_id))
                self.ia.append(self.h.create({'id_': device_id}))
                self.features.append(self.ia[-1].remote_device.node_map)
                self.default_settings(self.features[-1])
                self.data_settings(self.features[-1])
                print("Connected device: {}".format(device_id))
            self.connected = True
        
    def default_settings(self, features):

        # print(dir(self.features))
        features.PhotoneoTriggerMode.value = "Software"
        features.SendTexture.value = True
        features.SendPointCloud.value = True
        features.SendNormalMap.value = True
        features.SendDepthMap.value = True
    def data_settings(self, features):
        with open('config.json', 'r') as file:
            data = json.load(file)
        data = data.get("camera")
        features.CoordinateSpace.value = data['CoordinateSpace']
        features.RecognizeMarkers.value = eval(data['RecognizeMarkers'])
        features.CameraSpace.value = data['CameraSpace']
        features.NormalsEstimationRadius.value = int(data['NormalsEstimationRadius'])

    def results(self, index):
        return self.pcd[index], self.image[index], self.depth[index]


    def disconnect(self):
        if self.connected:
            self.ia.destroy()
            self.h.reset()
            self.h.remove_file(self.cti_file_path)
            self.connected = False

    def set_transformation_matrix(self, transformation_matrix, index):
        print("Saving estimated transformation matrix of device: {}".format(self.device_list[index]))
        self.set_rotation_matrix(transformation_matrix[:3, :3], index)
        self.set_translation_vector(transformation_matrix[:3, 3], index)
        print("Successfully saved estimated transformation matrix!")

    def set_translation_vector(self, translation_vector, index):
        robot_transformation_translation_vector_length = self.features[index].RobotTransformationTranslationVector.length
        robot_transformation_translation_vector_bytes = self.features[index].RobotTransformationTranslationVector.get(
            robot_transformation_translation_vector_length)
        robot_transformation_translation_vector = struct.unpack('3d', robot_transformation_translation_vector_bytes)
        robot_transformation_translation_vector_new_values = translation_vector.flatten().tolist()
        robot_transformation_translation_vector_new_bytes = struct.pack('3d',
                                                                        *robot_transformation_translation_vector_new_values)
        self.features[index].RobotTransformationTranslationVector.set(robot_transformation_translation_vector_new_bytes)

    def set_rotation_matrix(self, rotation_matrix, index):
        robot_transformation_rotation_matrix_length = self.features[index].RobotTransformationRotationMatrix.length
        robot_transformation_rotation_matrix_bytes = self.features[index].RobotTransformationRotationMatrix.get(robot_transformation_rotation_matrix_length)
        robot_transformation_rotation_matrix = struct.unpack('9d', robot_transformation_rotation_matrix_bytes)
        robot_transformation_rotation_matrix_new_values = rotation_matrix.flatten().tolist()
        robot_transformation_rotation_matrix_new_bytes = struct.pack('9d', *robot_transformation_rotation_matrix_new_values)
        self.features[index].RobotTransformationRotationMatrix.set(robot_transformation_rotation_matrix_new_bytes)
        return
    def trigger(self):
        if self.connected:
            for i in range(len(self.ia)):
                print("Triggering device: {}".format(self.device_list[i]))
                self.h.update
                # PhotoneoTL_DEV_<ID>


                # Order is fixed on the selected output structure. Disabled fields are shown as empty components.
                # Individual structures can enabled/disabled by the following features:
                # SendTexture, SendPointCloud, SendNormalMap, SendDepthMap, SendConfidenceMap, SendEventMap, SendColorCameraImage
                # payload.components[#]
                # [0] Texture
                # [1] TextureRGB
                # [2] PointCloud [X,Y,Z,...]
                # [3] NormalMap [X,Y,Z,...]
                # [4] DepthMap
                # [5] ConfidenceMap
                # [6] EventMap
                # [7] ColorCameraImage

                # Send every output structure

                #features.SendEventMap.value = True         # MotionCam-3D exclusive
                #features.SendColorCameraImage.value = True # MotionCam-3D Color exclusive

                self.ia[i].start()

                self.features[i].TriggerFrame.execute() # trigger frame
                with self.ia[i].fetch(timeout=10.0) as buffer:
                    payload = buffer.payload

                    texture_component = payload.components[0]
                    depth_component = payload.components[4]
                    self.depth[i] = self.display_depth_map_if_available(depth_component)
                    texture_rgb_component = payload.components[1]
                    self.display_color_image_if_available(texture_rgb_component)
                    color_image_component = payload.components[7]
                    self.image[i] = self.display_color_image_if_available(color_image_component)
                    point_cloud_component = payload.components[2]
                    norm_component = payload.components[3]
                    self.pcd[i] = self.display_pointcloud_if_available(point_cloud_component, norm_component, texture_component, texture_rgb_component)
                self.ia[i].stop()
                print("Successfully triggered device: {}".format(self.device_list[i]))

