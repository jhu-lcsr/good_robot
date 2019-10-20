"""
Forked from http://www.huyaoyu.com/technical/2018/06/23/convert-results-from-ros-camera_calibration-into-format-used-by-opencv.html
"""

from __future__ import print_function

# Description
# ===========
# 
# Convert the YAML files form the camera_calibration package in ROS into
# plain text files that coulde be used by OpenCV.
#
# Author
# ======
#
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
#
# Data
# ====
#
# Created 2018-06-22
# 

import os
import argparse

import numpy as np

from CameraInfo import CameraInfo

DISTORTION_COEFFICIENTS_LENGTH = 14

def load_camera_info(filename):
        """Load a yaml file as the camera info."""

        import yaml

        # Open the file.
        fp = open( filename, "r" )

        # Parse the yaml file.
        ci = yaml.safe_load(fp)

        # Copy the information into cameraInfo.
        cameraInfo = CameraInfo()

        cameraInfo.height = ci["image_height"]
        cameraInfo.width  = ci["image_width"]
        cameraInfo.distortion_model = ci["distortion_model"]
        cameraInfo.K = ci["camera_matrix"]["data"]
        cameraInfo.D = ci["distortion_coefficients"]["data"]
        cameraInfo.R = ci["rectification_matrix"]["data"]
        cameraInfo.P = ci["projection_matrix"]["data"]

        return cameraInfo

if __name__ == "__main__":
    # Create an argument parser.
    parser = argparse.ArgumentParser( description = "Read the yaml files that produced by the ROS node camera_calibration and convert them into files of OpenCV format." )
    parser.add_argument( "--from_dir",    nargs = "?", help="The directory where the yaml files are.")
    parser.add_argument( "--out_dir",     nargs = "?", help="The destination directory to put the converted files." )
    parser.add_argument( "--in_file_1", nargs = "?", default = "left.yaml", help="Input file of the left camera." )
    parser.add_argument( "--in_file_2", nargs = "?", default = "right.yaml", help="Input file of the right camera." )
    parser.add_argument( "--cm_1",      nargs = "?", default = "CameraMatrixRGB.dat", help="Output file of the RGB camera matrix, numpy txt format." )
    parser.add_argument( "--cm_2",      nargs = "?", default = "CameraMatrixDepth.dat", help="Output file of the Depth camera matrix, numpy txt format." )
    parser.add_argument( "--dst_1",     nargs = "?", default = "DistortionCoefficientRGB.dat", help="Output file of the RGB distortion coefficient, numpy txt format." )
    parser.add_argument( "--dst_2",     nargs = "?", default = "DistortionCoefficientDepth.dat", help="Output file of the Depth distortion coefficient, numpy txt format." )
    parser.add_argument( "--P_1",       nargs = "?", default = "ProjectionMatrixRGB.dat", help="Output file of the RGB projection matrix, numpy txt format." )
    parser.add_argument( "--P_2",       nargs = "?", default = "ProjectionMatrixDepth.dat", help="Output file of the Depth projection matrix, numpy txt format." )
    parser.add_argument( "--R_1",       nargs = "?", default = "RectificationMatrixRGB.dat", help="Output file of the RGB rotation/rectification matrix, numpy txt format." )
    parser.add_argument( "--R_2",       nargs = "?", default = "RectificationMatrixDepth.dat", help="Output file of the Depth rotation/rectification matrix, numpy txt format." )

    # Transfer the command line arguments to local variabless.
    args = parser.parse_args()

    fromDir = args.from_dir
    outDir  = args.out_dir

    inFiles = [ fromDir + "/" + args.in_file_1, fromDir + "/" + args.in_file_2 ]
    outCM   = [ outDir + "/" + args.cm_1,  outDir + "/" + args.cm_2 ]
    outDST  = [ outDir + "/" + args.dst_1, outDir + "/" + args.dst_2 ]
    outP    = [ outDir + "/" + args.P_1,   outDir + "/" + args.P_2 ]
    outR    = [ outDir + "/" + args.R_1,   outDir + "/" + args.R_2 ]

    # Read the yaml files.
    cameraInfo = []

    cameraInfo.append( load_camera_info(inFiles[0]) )
    cameraInfo.append( load_camera_info(inFiles[1]) )

    # Test the destination directory.
    if ( False == os.path.isdir( outDir ) ):
        print("The destination directory (%s) does not exist. Create the directory." % outDir)
        os.mkdir( outDir )

    # Output data to file system.
    nCameraInfo = len( cameraInfo )

    for i in range(nCameraInfo):
        print("Process the %d/%dth yaml file..." % ( i+1, nCameraInfo ))

        ci = cameraInfo[i]

        # Camera matrix.
        cameraMatrix = np.array( ci.K ).reshape((3, 3))
        np.savetxt( outCM[i], cameraMatrix )

        # Distortion coefficients.
        distortionCoefficient = np.zeros((1, DISTORTION_COEFFICIENTS_LENGTH))
        nDSTYaml = len( ci.D )
        distortionCoefficient[0, :nDSTYaml] = ci.D
        np.savetxt( outDST[i], distortionCoefficient )

        # Projection matrix.
        P = np.array( ci.P ).reshape((3, 4))
        np.savetxt( outP[i], P )

        # Rotation/Rectification matrix.
        R = np.array( ci.R ).reshape((3, 3))
        np.savetxt( outR[i], R )

    print("Done.")