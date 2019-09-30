'''
Alan Perception Module.
Authors: Jeff, Jacky
'''
import logging

from .version import __version__
from .camera_intrinsics import CameraIntrinsics
from .orthographic_intrinsics import OrthographicIntrinsics
from .exceptions import SensorUnresponsiveException

try:
    from .cnn import AlexNet, AlexNetWeights, conv
    from .features import Feature, LocalFeature, GlobalFeature, SHOTFeature, MVCNNFeature, BagOfFeatures
    from .feature_extractors import FeatureExtractor, CNNBatchFeatureExtractor, CNNReusableBatchFeatureExtractor
except Exception:
    logging.warning('Unable to import CNN modules! Likely due to missing tensorflow.')
    logging.warning('TensorFlow can be installed following the instructions in https://www.tensorflow.org/get_started/os_setup')

from .feature_matcher import Correspondences, NormalCorrespondences, FeatureMatcher, RawDistanceFeatureMatcher, PointToPlaneFeatureMatcher
from .image import Image, ColorImage, DepthImage, IrImage, GrayscaleImage, RgbdImage, GdImage, SegmentationImage, BinaryImage, PointCloudImage, NormalCloudImage
from .object_render import RenderMode, ObjectRender, QueryImageBundle
from .chessboard_registration import ChessboardRegistrationResult, CameraChessboardRegistration
from .point_registration import RegistrationResult, IterativeRegistrationSolver, PointToPlaneICPSolver
from .detector import RgbdDetection, RgbdDetector, RgbdForegroundMaskDetector, RgbdForegroundMaskQueryImageDetector, PointCloudBoxDetector, RgbdDetectorFactory
from .camera_sensor import CameraSensor, VirtualSensor, TensorDatasetVirtualSensor
from .webcam_sensor import WebcamSensor

try:
    from .kinect2_sensor import Kinect2PacketPipelineMode, Kinect2FrameMode, Kinect2RegistrationMode, Kinect2DepthMode, Kinect2BridgedQuality, Kinect2Sensor, KinectSensorBridged, VirtualKinect2Sensor, Kinect2SensorFactory, load_images
except Exception:
    logging.warning('Unable to import Kinect2 sensor modules! Likely due to missing pylibfreenect2.')
    logging.warning('The pylibfreenect2 library can be installed from https://github.com/r9y9/pylibfreenect2')

try:
    from .primesense_sensor import PrimesenseSensor, PrimesenseSensor_ROS, PrimesenseRegistrationMode
except Exception:
    logging.warning('Unable to import Primsense sensor modules! Likely due to missing OpenNI2.')

try:
    from .realsense_sensor import RealSenseSensor
except Exception:
    logging.warning('Unable to import RealSense sensor modules!')

try:
    from .ensenso_sensor import EnsensoSensor
except Exception:
    logging.warning('Unable to import Ensenso sensor modules!.')

try:
    from .phoxi_sensor import PhoXiSensor
    from .colorized_phoxi_sensor import ColorizedPhoXiSensor
except Exception as e:
    logging.warning('Unable to import PhoXi sensor modules!')

try:
    from .opencv_camera_sensor import OpenCVCameraSensor
    from .rgbd_sensors import RgbdSensorFactory
except Exception:
    logging.warning('Unable to import generic sensor modules!.')

try:
    from .weight_sensor import WeightSensor
except:
    logging.warning('Unable to import weight sensor modules!')

from .video_recorder import VideoRecorder

__all__ = [
    'CameraIntrinsics',
    'AlexNetWeights', 'AlexNet', 'conv',
    'RgbdDetection', 'RgbdDetector', 'RgbdForegroundMaskDetector', 'RgbdForegroundMaskQueryImageDetector', 'PointCloudBoxDetector', 'RgbdDetectorFactory',
    'FeatureExtractor', 'CNNBatchFeatureExtractor', 'CNNReusableBatchFeatureExtractor',
    'Correspondences', 'NormalCorrespondences', 'FeatureMatcher', 'RawDistanceFeatureMatcher', 'PointToPlaneFeatureMatcher',
    'Feature', 'LocalFeature', 'GlobalFeature', 'SHOTFeature', 'MVCNNFeature', 'BagOfFeatures',
    'Image', 'ColorImage', 'DepthImage', 'IrImage', 'GrayscaleImage', 'RgbdImage', 'GdImage', 'SegmentationImage', 'BinaryImage', 'PointCloudImage', 'NormalCloudImage',
    'Kinect2PacketPipelineMode', 'Kinect2FrameMode', 'Kinect2RegistrationMode', 'Kinect2DepthMode', 'Kinect2BridgedQuality', 'Kinect2Sensor','KinectSensorBridged','VirtualKinect2Sensor', 'Kinect2SensorFactory', 'load_images',
    'EnsensoSensor',
    'RgbdSensorFactory', 'PrimesenseSensor', 'VirtualPrimesenseSensor', 'PrimesenseSensor_ROS', 'PrimesenseRegistrationMode',
    'RealSenseSensor',
    'RenderMode', 'ObjectRender', 'QueryImageBundle',
    'RegistrationResult', 'IterativeRegistrationSolver', 'PointToPlaneICPSolver',
    'OpenCVCameraSensor',
    'VideoRecorder',
]
