"""
Function Name: BarcodeEncoder

Description: this part is not finish yet. but for the finally it will be added into the main

Argument:

Parameters:

Return:
    [type] -> [description]

Edited by: [date] [author name]
"""
from pyzbar import pyzbar
import cv2

class BarcodeEncoder(object):

    def __init__(self):
        """
        Function Name: __init__
        
        Description: Barcode encoder
        
        Argument:
        
        Parameters:
        
        Return:
        
        Edited by: [2020-10-14] [Pawat]
        """        
        pass

    def execute(self, image):

        barcodes = pyzbar.decode(image)

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            barcodeData = barcode.data.decode('utf-8')
            barcodeType = barcode.type
            text = "{} ( {} )".format(barcodeData, barcodeType)

            # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            print("Information : \n Found Type : {} Barcode : {}".format(barcodeType, barcodeData))

        return barcodeData, barcodeType

    