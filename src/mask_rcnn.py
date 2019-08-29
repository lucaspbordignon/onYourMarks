import cv2 as cv
import argparse
import numpy as np
import os.path
import sys

import wget
import tarfile

from functions import Utils

# Model Constants
OUTPUT_PATH = 'mask_rcnn_output'
WEIGHTS_PATH = '../model/mask_rcnn_inception_v2_coco_2018_01_28/' \
               'frozen_inference_graph.pb'
TF_WEIGHTS_GRAPH_PATH = '../model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
WEIGHTS_URL = 'http://download.tensorflow.org/models/object_detection/mask_' \
              'rcnn_inception_v2_coco_2018_01_28.tar.gz'

DISPARITY_PATH = './disparity_map.avi'

# Manual entries for the model. Run with -h flag to use it
parser = argparse.ArgumentParser(description='Use this script to run '
                                             'Mask-RCNN object detection '
                                             'and segmentation')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--kitti', help='To use kitti images.', default=False)
parser.add_argument('--disparity', help='Path to disparity map video file.')
args = parser.parse_args()

# Download and extract weight files from Mask-RCNN on Coco
if (not os.path.exists(WEIGHTS_PATH)):
    weights_file = wget.download(WEIGHTS_URL)

    tar = tarfile.open(weights_file)
    tar.extractall(path='../model/')


# Load the network from Tensorflow Model
net = cv.dnn.readNetFromTensorflow(WEIGHTS_PATH, TF_WEIGHTS_GRAPH_PATH)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)


# Setup the output view
cv.namedWindow('Extracted Features', cv.WINDOW_NORMAL)

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)

    cap = cv.VideoCapture(args.image)
    outputFile = OUTPUT_PATH + args.image[:-4] + '.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)

    cap = cv.VideoCapture(args.video)
    outputFile = OUTPUT_PATH + args.video[:-4] + '.avi'
elif (args.kitti):
    # ipen kitti sample file
    if not os.path.isfile(args.kitti):
        print("Input kitti file doesn't exist")
        sys.exit(1)

    cap = cv.VideoCapture('KITTI_Germany/City/City/2011_09_26_6/'
                          'image_02/data/%10d.png')
    outputFile = OUTPUT_PATH + '.avi'
else:
    cap = cv.VideoCapture(0)  # Internal Camera input

    outputFile = OUTPUT_PATH + '.avi'

    # Get disparitymap frames (or video) input
    capDisparity = cv.VideoCapture(args.disparity or DISPARITY_PATH)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile,
                                cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                15,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

width = cap.get(3)  # P/ OF
height = cap.get(4)  # P/ OF

ret, prev = cap.read()
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

processor = Utils()

while cv.waitKey(1) < 0:
    # Get frame from the video
    hasFrame, frame = cap.read()
    hasImghsv, imghsv = capDisparity.read()

    # Stop the program if reached end of video
    if not hasFrame or not hasImghsv:
        print("Done processing! Output is stored as {}".format(outputFile))
        cv.waitKey(3000)
        break

    # Optical Flow calculation
    # P / OF
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    # END P/ OF

    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Extract the bounding box and mask for each of the detected objects
    processor.extract_segments(frame, boxes, masks, imghsv, flow, gray)

    # Put efficiency information.
    t, _ = net.getPerfProfile()

    print('[INFO] Mask-RCNN on Jetson TX2. Inference time: '
          '%0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency()))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow('Extracted Features', frame)
