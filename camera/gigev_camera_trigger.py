
import sys, os
sys.path.append(os.getcwd()+'/camera')
from pathlib import Path
import cv2
from genicam.genapi import NodeMap
from harvesters.core import Harvester
import open3d as o3d
from gentl_producer_loader import producer_path
import json
import numpy as np
from photoneo_genicam.components import enable_components, enabled_components
from photoneo_genicam.pointcloud import create_3d_vector, map_texture

class GIGEV_Device:
    def __init__(self, device_id):
        self.device_id = device_id
        self.connected = False
        self.pcd = None
        self.depth = None
        self.image = None

    def display_depth_map_if_available(self, image):
        color_image = cv2.cvtColor(image.data.reshape((image.height, image.width, 3)), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    def display_color_image_if_available(self, img):
        return cv2.cvtColor(img.data.reshape((img.height, img.width, 3)), cv2.COLOR_RGB2BGR)

    def display_pointcloud_if_available(self, buffer):
        components = dict(zip(enabled_components(self.features), buffer.payload.components))
        intensity_component: Component2DImage = components["Intensity"]
        point_cloud_raw: Component2DImage = components["Range"]
        normal_component: Component2DImage = components["Normal"]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = create_3d_vector(point_cloud_raw.data)
        point_cloud.normals = create_3d_vector(normal_component.data)
        point_cloud.colors = map_texture(intensity_component)
        return point_cloud

    def connect(self):
        print("Connecting device: {}".format(self.device_id))
        if not self.connected:
            self.h = Harvester()
            self.h.add_file(str(producer_path), check_existence=True, check_validity=True)
            self.h.update()
            self.connected = True
            print("Connected device: {}".format(self.device_id))

        
    def default_settings(self, ia):
        self.features : NodeMap = ia.remote_device.node_map
        self.features.TriggerSelector.value = "FrameStart"
        self.features.TriggerMode.value = "On"
        self.features.TriggerSource.value = "Software"
        self.features.ChunkModeActive.value = True
        self.features.CameraTextureSource.value = "Color"
        self.features.PixelFormat.value = "RGB8"
        self.features.Scan3dOutputMode.value = "CalibratedABC_Grid"

    def data_settings(self):
        with open('config.json', 'r') as file:
            data = json.load(file)
        data = data.get("camera")
        self.features.OperationMode.value = data['OperationMode']
        self.features.CoordinateSpace.value = data['CoordinateSpace']
        self.features.RecognizeMarkers.value = eval(data['RecognizeMarkers'])
        self.features.CameraSpace.value = data['CameraSpace']
        self.features.NormalsEstimationRadius.value = int(data['NormalsEstimationRadius'])



    def disconnect(self):
        if self.connected:
            self.h.reset()
            self.h.remove_file(str(producer_path))
            self.connected = False

    def trigger(self):
        if self.connected:
            print("Triggering device: {}".format(self.device_id))
            with self.h.create({'serial_number': self.device_id}) as ia:
                self.default_settings(ia)
                self.data_settings()
                enable_components(self.features, ["Intensity", "Range", "Normal"])
                self.h.update
                ia.start()
                self.features.TriggerSoftware.execute() # trigger frame
                with ia.fetch(timeout=10.0) as buffer:
                    self.image = self.display_color_image_if_available(buffer.payload.components[0])
                    self.depth = self.display_depth_map_if_available(buffer.payload.components[1])
                    self.pcd = self.display_pointcloud_if_available(buffer)
                    assert result_image.shape[0]*result_image.shape[1] == np.asarray(result_pcd.points).shape[0]
                    assert result_image.shape[0]*result_image.shape[1] == result_depth.shape[0]*result_depth.shape[1]
            print("Successfully triggered device: {}".format(self.device_id))