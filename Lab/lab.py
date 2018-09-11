# from openni import openni2

# openni2.initialize()

# dev = openni2.Device.open_any()

# depth_stream = dev.create_depth_stream()
# depth_stream.start()
# frame = depth_stream.read_frame()
# frame_data = frame.get_buffer_as_uint16()
# depth_stream.stop()

# openni2.unload()

import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
openni2.initialize()
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))