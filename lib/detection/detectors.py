"""
Function Name: 

Description: 
    A collection of detectors, including sticker detector, text detector...

Argument:  
            
Parameters: None

Return: None           
Edited by: [2020/08/26] [Vincent Wang]
"""
from pathlib import Path
import copy
import time
from cv2 import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression
# from ..core.utils import Logger, Timer

class TextDetectorPerfectBox(object):
    def __init__(self, opt: dict):
        """
        Function Name: __init__
        
        Description: Text detector using EAST model. See more information in
        https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

        Argument:
            opt [dict] -> Options for text detector. See config.json for more info.
        
        Parameters:
        
        Return:
        
        Edited by: [2020-10-14] [Pawat]
        """        
        # self._log = Logger.get("TextDetectorPerfectBox")
        self._net = cv.dnn.readNet(str(opt.model_path))
        self._layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

    @staticmethod
    def decodePredictions(scores, geometry):
        """
        Function Name: decodePredictions
        
        Description: To calculate bounding box from scores and geometry
        
        Argument:
            scores [[type]] -> Confident of the box
            geometry [[type]] -> Geometry is coordinate or data of the box before computing
        
        Parameters:
        
        Return:
            [tuple of two array] -> (rects, confidences)
        
        Edited by: [2020-10-14] [Pawat]
        """         
            
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):

            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):

                if scoresData[x] < 0.5:
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                end_x = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                end_y = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(scoresData[x])

        return (rects, confidences)

    def execute(self, img: np.ndarray):
        """
        Function Name: execute
        
        Description: Execute text detection
        
        Argument:
            img (np.ndarray): Image to detect
        
        Parameters:
        
        Return:
            list -> List of bounding boxes, starting point and ending point of the boxes and delta x and delta y 
                    in form  [start_x, start_y, end_x, end_y, del_x, del_y]
        
        Edited by: [2020-10-14] [Pawat]
        """        
        
        (W, H) = (None, None)
        (newW, newH) = (320, 320)
        (rW, rH) = (None, None)

        if W is None or H is None:
            (H, W) = img.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
        
        img = cv.resize(img, (newW, newH))
        t0 = time.time()
        blob = cv.dnn.blobFromImage(img, 1.0, (newW, newH),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
        t1 = time.time()
        # print("blob : ",t1-t0)
        self._net.setInput(blob)
        t2 = time.time()
        # print("set input : ", t2-t1)
        (scores, geometry) = self._net.forward(self._layer_names)
        t3 = time.time()
        # print("forward process: ",t3-t2)
        (rects, confidences) = self.decodePredictions(scores, geometry)
        t4 = time.time()
        # print("decode : ",t4-t3)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        sort_box = sorted(boxes, key=lambda s: s[1])
        all_boxes_data = []
        for i, (start_x, start_y, end_x, end_y) in enumerate(sort_box):
            start_x = int(start_x * rW)
            start_y = int(start_y * rH)
            end_x = int(end_x * rW)
            end_y = int(end_y * rH)
            del_x = end_x - start_x
            del_y = end_y - start_y
            all_boxes_data.append(
                [start_x, start_y, end_x, end_y])
        return all_boxes_data



    def __call__(self, img: np.ndarray) :
        """
        Function Name: __call__
        
        Description: execute(img)
        
        Argument:
            img [np.ndarray] -> image for Testdetection
        
        Parameters:
        
        Edited by: [2020-10-14] [Pawat]
        """

        return self.execute(img)

class ArrangeBox(object):

    def __init__(self, opt):
        """
        Function Name: __init__
        
        Description: to merge many small boxes in the same line to be one line(to cover all of the line)
                    also the many small columns those will be merge to be the same big box. But in both merging line and column
                    if the gap is larger than threshold (be set in json files) it will not be merged. 
                    equation of thresholding 
                        Line critrion 
                        Reference : start point of y axis
                         merge < (delta y of the box) * factor_line < not merge 

                        Column critrion
                        Reference : start point of x axis 
                        merge < (delta x of the box) + (delta y of the box) * factor_line < not merge

        
        Argument:
            opt [list] -> [To input column_factor and line_factor]
        
        Parameters:
        
        Return:
        
        Edited by: [2020-10-14] [Pawat]
        """        
        self._opt = opt
        pass

    def execute(self, all_boxes_data):
        """
        Function Name: Proceed->  merging many small boxes in the same line to be one line(to cover all of the line)
                    also the many small columns those will be merge to be the same big box. But in both merging line and column
                    if the gap is larger than threshold (be set in json files) it will not be merged. 
                    equation of thresholding 
                        Line critrion 
                        Reference : start point of y axis
                         merge < (delta y of the box) * factor_line < not merge 

                        Column critrion
                        Reference : start point of x axis 
                        merge < (delta x of the box) + (delta y of the box) * factor_line < not merge
        
        Description: [summary]
        
        Argument:
            all_boxes_data [list] -> the list of boxes by testdetection 
        
        Parameters:
        
        Return:
            [list] -> box of perfectly rearrange the content is included coordinate of top-right anf bottom-left
                        in this form  [start_x, start_y, end_x, end_y]
        
        Edited by: [2020-10-14] [Pawat]
        """        

        sort_sy_all_boxes_data = sorted(all_boxes_data, key=lambda s: s[1])
        find_lines = []

        cut = 0
        for j, [start_x, start_y, end_x, end_y, del_x, del_y] in enumerate(sort_sy_all_boxes_data):

            if abs(sort_sy_all_boxes_data[j][1] - sort_sy_all_boxes_data[j - 1][1]) < del_y * self._opt.line_factor:
                if j == len(sort_sy_all_boxes_data) - 1:
                    find_lines.append(sort_sy_all_boxes_data[cut:j + 1])
                else:
                    pass
            else:
                if j != len(sort_sy_all_boxes_data) - 1:
                    find_lines.append(sort_sy_all_boxes_data[cut:j])
                    cut = j
                elif j == len(sort_sy_all_boxes_data) - 1:
                    find_lines.append(sort_sy_all_boxes_data[cut:j])
                    find_lines.append(
                        [[start_x, start_y, end_x, end_y, del_x, del_y]])
        cut_text = 0
        text_in_line = []
        all_texts = []
        if find_lines != []:
            for line in find_lines:
                if len(line) != 0:
                    line = sorted(line, key=lambda s: s[0])
                    if len(line) == 1:
                        text_in_line = [line]
                    else:
                        for k, [start_x, start_y, end_x, end_y, del_x, del_y] in enumerate(line):
                            if abs(line[k][0] - line[k - 1][0]) < del_x + del_y * self._opt.column_factor:
                                if k == len(line) - 1:
                                    text_in_line.append(line[cut_text:k + 1])
                                else:
                                    pass
                            else:
                                if k != len(line) - 1:
                                    text_in_line.append(line[cut_text:k])
                                    cut_text = k
                                elif k == len(line) - 1:
                                    text_in_line.append(line[cut_text:k])
                                    text_in_line.append(
                                        [[start_x, start_y, end_x, end_y, del_x, del_y]])
                else:
                    pass
                if text_in_line != []:
                    all_texts.append(text_in_line)
                    text_in_line = []

        complete_box = []
        if all_texts != []:

            for p in range(len(all_texts)):

                for k, line in enumerate(all_texts[p]):

                    if len(all_texts[p][k]) > 1:
                        sort_line = np.sort(all_texts[p][k], axis=0)
                        [start_x0, start_y0, end_x0, end_y0,
                         del_x0, del_y0] = sort_line[0]
                        [start_x1, start_y1, end_x1, end_y1,
                         del_x1, del_y1] = sort_line[-1]
                        complete_box.append(
                            [start_x0, start_y0, end_x1, end_y1, abs(start_x0 - end_x0), abs(start_y1 - end_y1)])

                    elif len(all_texts[p][k]) == 1:
                        [start_x0, start_y0, end_x0, end_y0,
                         del_x0, del_y0] = line[0]
                        complete_box.append(
                            [start_x0, start_y0, end_x0, end_y0, abs(start_x0 - end_x0), abs(start_y0 - end_y0)])

        c_sort_1 = sorted(complete_box, key=lambda s: s[0])
        get_column = []
        separate_column = []
        cut_column = 0

        for r, [start_x, start_y, end_x, end_y, del_x, del_y] in enumerate(c_sort_1):

            if abs(c_sort_1[r][0] - c_sort_1[r - 1][0]) < 1.4 * del_x:

                if r == len(c_sort_1) - 1:
                    get_column.append(c_sort_1[cut_column: r + 1])
                else:
                    pass
            else:
                if r != len(c_sort_1) - 1:
                    get_column.append(c_sort_1[cut_column:r])
                    cut_column = r
                    if cut_column != 0:
                        separate_column.append(cut_column)
                elif r == len(c_sort_1) - 1:
                    get_column.append(c_sort_1[cut_column:r])
                    get_column.append(
                        [[start_x, start_y, end_x, end_y, del_x, del_y]])

        complete_box_column = []
        for i in range(len(get_column)):
            if get_column[i] != []:
                sort_column = sorted(get_column[i], key=lambda s: s[2])
                min_sort_col = np.min(sort_column, axis=0)
                max_sort_col = np.max(sort_column, axis=0)
                for j, column in enumerate(sort_column):
                    if column != []:
                        column[0] = min_sort_col[0]
                        column[2] = max_sort_col[2] + max_sort_col[5]

                        complete_box_column.append(column[0:4])

        
        return complete_box_column

    def delete_duplicate(self):
        pass

    def __call__(self, boxes):
        return self.execute(boxes)


