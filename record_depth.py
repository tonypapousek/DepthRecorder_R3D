#!/usr/bin/env python3

# Based on code found @https://github.com/marek-simonik/record3d

import numpy as np # Filtering, concatenation, matrices and more
from record3d import Record3DStream # Capture data from iPhone
import cv2 # Colorspace conversion
from threading import Event # Events, timing
# import ffmpeg # Video encoding
from vidgear.gears import WriteGear # Video writing
import time # Dynamic filename
import json # Saving metadata for point clouds
from pathlib import Path # Folder/file handling

REPLACEMENT_DEPTH = np.nan # Replacement value for depth filtering
MAX_DEPTH = 3 # Everything after MAX_DEPTH is replaced with REPLACEMENT_DEPTH

DEBUG_MODE = False
ADVANCED_DEBUG = False

# Aim for speed > 1.x. Slower is fine (it eventually encodes), but the UX suffers

CODEC_PARAMS = {"-vcodec": "libx264", "-crf": 0, "-preset": "veryfast"}

class DepthRecorder:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.first_frame = True
        self.has_stopped = False


    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

        self.has_stopped = True

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self):
        timestamp = int(time.time())
        self.folder = "output" + "/" + str(timestamp)
        self.file_name = "merged.mp4"
        self.file_path = self.folder + "/" + self.file_name
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        writer = WriteGear(output_filename=self.file_path, compression_mode=True, logging=True, **CODEC_PARAMS)

        while True:
            self.event.wait()  # Wait for new frame to arrive
            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            bgr = self.session.get_rgb_frame()

            self.intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat()) # Get metadata for appending to file
            bgr = cv2.normalize(cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if (MAX_DEPTH):
                depth[depth > MAX_DEPTH] = REPLACEMENT_DEPTH # Filter out unwanted background details
            depth_hsv = depth_to_hsv(depth)
            depth_bgr = cv2.cvtColor(depth_hsv, cv2.COLOR_HSV2BGR_FULL)
            merged = np.matrix = np.concatenate((depth_bgr, bgr), axis=1)
            frame = merged * 255
            writer.write(frame.astype(np.uint8)) # Write current frame           
            # print("displaying stream...")

            # Show the RGBD Stream
            # cv2.imshow('RGB', rgb)
            #cv2.imshow('Depth', depth)
            #cv2.imshow('Depth HSV', depth_hsv)
            # cv2.imshow('Depth RGB', depth_rgb)
            cv2.imshow('Merged', merged)
            # cv2.imshow('Frame', frame)

            cv2.waitKey(1)

            if (self.first_frame):
                self.first_frame = False
                self.frame_height = merged.shape[0]
                self.frame_width = merged.shape[1]
                if(DEBUG_MODE):
                    print("Resolution: ", str(self.frame_width) + "x" + str(self.frame_height))
                if(ADVANCED_DEBUG):
                    print("depth frame: ", depth)
                    print("depth hsv: ", depth_hsv)
                    print("merged:", merged)

            if (self.has_stopped):
                writer.close()
                cv2.destroyAllWindows() 
        
                if(DEBUG_MODE):
                    print("mat:", self.intrinsic_mat)

                # Prepare metadata for end of file
                metadata = {"intrinsicMatrix": self.intrinsic_mat.flatten().tolist()}
                if(DEBUG_MODE):
                    print("metadata:", metadata)
                # The default http://record3d.xyz/ viewer fails to parse the metadata without some extra character.
                # It uses length-1 instead of just length. Padding the output appears to make things work.
                metaString = json.dumps(metadata) + " " 
                if(DEBUG_MODE):
                    print("metaString", metaString) # For concatenation e.g. "{\"intrinsicMatrix\":[593.76177978515625,0,0,0,594.7872314453125,0,241.74200439453125,319.98410034179688,1]}"

                metaFile = Path(self.folder + "/meta.json") # Useful to have for further video editing. Append at end of .mp4 file for record3d player
                metaFile.write_text(metaString)

                videoFile = Path(self.file_path).open("a") # Only append the metadata
                videoFile.write(metaString)
                videoFile.close()

                break

            self.event.clear()

# Convert 32-bit depth matrices to HSV image
# From Marek @ Record3D: 
#   "Record3D simply clamps the depth to maximum of 3 meters, divides the depth value by 3,
#    so that the depth values are normalized into the 0–1 range and then treats the normalized
#    value as the Hue component in HSV color space. The HSV color then gets converted into RGB.
#    The process is described on the Github page from my previous email.
#    You will have to create a new empty RGB image in OpenCV and populate it by converting 32bit depth > Hue > HSV > RGB."
# @see https://github.com/marek-simonik/record3d-simple-wifi-streaming-demo#22-rgbd-video-format
# Note: Depth matrix contains 32 bit floats, with values 1-3 (meters) E.g.
# [[      nan       nan       nan ... 1.8025001 1.8075001 1.8075001]
# [      nan       nan       nan ... 1.8025001 1.8075001 1.8075001]
# [      nan       nan       nan ... 1.8025001 1.8075001 1.8075001]
# ...
# [2.053625  2.047125  2.03425   ... 2.053625  2.07325   2.086625 ]
# [2.060125  2.053625  2.047125  ... 2.066625  2.079875  2.100125 ]
# [2.066625  2.060125  2.053625  ... 2.086625  2.100125  2.1138752]]
def depth_to_hsv(depth: list[list[float]]):
    hue = (depth / 3) * 180 # Normalize range
    saturation = np.full_like(hue, 1)
    value = saturation
    hsv = cv2.merge((hue, saturation, value))
    return hsv

if __name__ == '__main__':
    app = DepthRecorder()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()