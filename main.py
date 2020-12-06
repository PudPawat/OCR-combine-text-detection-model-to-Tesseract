
import json
from easydict import EasyDict
from cv2 import cv2 as cv
import numpy as np
import pytesseract
from pathlib import Path
from lib.detection import TextDetectorPerfectBox

class Main:
    def __init__(self, opt):

        self._opt = opt
        self._source = opt.basic.source if "source" in opt.basic else 0
        self._fps = opt.basic.fps if "fps" in opt.basic else 30
        self._width = opt.basic.width if "width" in opt.basic else 640
        self._height = opt.basic.height if "height" in opt.basic else 480
        self._listpath = opt.basic.listpath if "listpath" in opt.basic else "docs/list.json"
        self._thick_text = opt.basic.puttext.thickness
        self._ratio_text = opt.basic.puttext.ratio

    def main(self):

        cap = cv.VideoCapture(self._source, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv.CAP_PROP_FPS, self._fps)

        if cap.isOpened() is False:
            # self._log.fatal("Cannot open video source \"%s\"", self._source)
            print("Cannot open video source \"%s\"", self._source)

        else:
            text_det = TextDetectorPerfectBox(self._opt.dl.text_detect)

            while True:
                _, img = cap.read()

                boxes = text_det(img)
                print(boxes)
                if boxes != []:
                    for z, (start_x, start_y, end_x, end_y) in enumerate(boxes):
                        # if self._opt.basic.show_line_box == "on":
                        cv.rectangle(img,
                                     (start_x, start_y),
                                     ( end_x, end_y),
                                     (255, 255, 0),
                                     thickness=1)
                        del_y = end_y - start_y
                        del_x = end_x - start_x
                        crop = img[int(np.clip(start_y - (-0.05)*del_y, 0, self._height)):int(
                            np.clip(end_y + 0.2*del_y, 0, self._height)),
                               int(np.clip(start_x - 0*del_x, 0, self._width)):int(
                                   np.clip(end_x + 0.05*del_x, 0, self._width))]
                        text = pytesseract.image_to_string(
                            crop, lang='eng')  # config=custom_config
                        cv.putText(img,text,(start_x,end_y),cv.FONT_HERSHEY_SIMPLEX,
                               self._ratio_text,
                               (0, 255, 0),
                               thickness=self._thick_text)

                cv.imshow("final",img)
                cv.waitKey(1)

if __name__ == "__main__":

    with Path("config/config.json").open("r") as f:
        config = json.load(f)
    config = EasyDict(config)
    Main(config).main()
