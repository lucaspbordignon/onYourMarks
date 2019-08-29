import cv2 as cv
import numpy as np

# Initialize the parameters
maskThreshold = 0.3  # Mask threshold
confThreshold = 0.5  # Confidence threshold


class Utils():
    CLASSES_PATH = '../model/mscoco_labels.names'
    COLORS_PATH = '../model/colors.txt'

    def __init__(self):
        self.load_colors()
        self.load_classes()

    def load_colors(self):
        self.colors = []

        with open(self.COLORS_PATH, 'rt') as f:
            colorsStr = f.read().rstrip('\n').split('\n')

        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(' ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])

            self.colors.append(color)

    def load_classes(self):
        self.classes = None

        with open(self.CLASSES_PATH, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def drawBox(self,
                frame,
                classId,
                conf,
                left,
                top,
                right,
                bottom,
                classMask,
                imghsv,
                flow,
                gray):
        '''
            Draw the predicted bounding box, colorize and show the mask on
            the image
        '''

        color = self.colors[classId % len(self.colors)]

        # Draw a bounding box
        cv.rectangle(frame, (left, top), (right, bottom), color, 1)

        # Print a label of class.
        label = (self.classes[classId])

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN,
                                             0.65, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame,
                     (left, top - round(labelSize[1])),
                     (left + round(labelSize[0]), top + baseLine),
                     (255, 255, 255),
                     cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1)

        # Resize the mask, threshold, color and apply it on the image
        classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
        mask = (classMask > maskThreshold)

        # Draw mask with Dispatry values
        selected_frame = frame[top:bottom+1, left:right+1][mask]
        selected_hsv = imghsv[top:bottom+1, left:right+1][mask]
        modified_frame = cv.addWeighted(selected_frame, 0.3,
                                        selected_hsv, 0.7, 0)

        frame[top:bottom + 1, left:right + 1][mask] = modified_frame

        # --------------------------------------------- #
        # ------ Disparity Map Average Intensity ------ #
        # --------------------------------------------- #
        grayRoi = cv.cvtColor(imghsv, cv.COLOR_BGR2GRAY)

        # Contours on the image
        mask = mask.astype(np.uint8)
        im2, contours, hierarchy = cv.findContours(mask,
                                                   cv.RETR_TREE,
                                                   cv.CHAIN_APPROX_SIMPLE)

        # Calculate the distances to the contour
        raw_dist = np.empty(mask.shape, dtype=np.float32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                raw_dist[i, j] = cv.pointPolygonTest(contours[0], (j, i), True)

        # Depicting the distances graphically
        veCount = 0
        intensityCount = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if raw_dist[i, j] >= 0:
                    intensity = grayRoi[top + i, left + j]
                    veCount = veCount + 1
                    intensityCount = intensityCount + intensity

        # ---------------------------------------- #
        # ---------- END INTENSITY --------------- #
        # ---------------------------------------- #

        # ----------------------------------------------------- #
        # - OpticalFlow in the object average and centralized - #
        # ----------------------------------------------------- #
        step = 16
        ofCount = 0
        fxCount = 0
        fyCount = 0
        for y in range(top, bottom, step):
            for x in range(left, right, step):
                fx, fy = flow[y, x].T
                ofCount = ofCount + 1
                fxCount = fxCount + fx
                fyCount = fyCount + fy

        fxM = fxCount/ofCount
        fyM = fyCount/ofCount

        # flowVectorLength = fxM * fyM
        # print(" --> Flow Length --> ", abs(flowVectorLength))

        objCenterW = int(round((left+right)/2))
        objCenterH = int(round((top+bottom)/2))

        cv.line(frame,
                (objCenterW, objCenterH),
                (int(round(objCenterW + fxM)), int(round(objCenterH + fyM))),
                (255, 0, 0),
                1)
        cv.circle(frame,
                  (int(round(objCenterW + fxM)), int(round(objCenterH + fyM))),
                  1,
                  (255, 0, 0),
                  -1)
        # ---------------------------------------------------------------- #
        # ------ OF ends in the object with average and centralized ------ #
        # ---------------------------------------------------------------- #

    def extract_segments(self, frame, boxes, masks, imghsv, flow, gray):
        '''
            For each frame, extract the bounding box and mask for
            each detected object

            # Output size of masks is NxCxHxW where
            # N - number of detected boxes
            # C - number of classes (excluding background)
            # HxW - segmentation shape
        '''
        numDetections = boxes.shape[2]

        frameH = imghsv.shape[0]
        frameW = imghsv.shape[1]

        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > confThreshold:
                classId = int(box[1])

                # Extract the bounding box
                left = int(frameW * box[3])
                top = int(frameH * box[4])
                right = int(frameW * box[5])
                bottom = int(frameH * box[6])

                left = max(0, min(left, frameW - 1))
                top = max(0, min(top, frameH - 1))
                right = max(0, min(right, frameW - 1))
                bottom = max(0, min(bottom, frameH - 1))

                # Extract the mask for the object
                classMask = mask[classId]

                # Draw bounding box, colorize and show the mask on the image
                self.drawBox(frame, classId, score,
                             left, top, right,
                             bottom, classMask, imghsv,
                             flow, gray)
