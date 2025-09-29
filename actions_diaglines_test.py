import numpy as np
from skimage.transform import probabilistic_hough_line
import math
import cv2

import os
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.morphology import skeletonize
from skimage.color import rgba2rgb, rgb2gray


class Config:
    def __init__(self):
        # actions / wireDetect config
        self.threshold = 0.05  # default: 0.05 /// % of empty space required in region either side of a wire for it to count. Value range between 0 and 1
        # actions / wireScanHough config
        self.minWireLength = 10  # default 10 /// minimum length a line can be for it to be classed as a wire
        self.borderSize = 15  # default 15  /// how many pixels the inspection region extends either side of the wire
        # actions / objectDetection config
        self.maximumDistance = 85  # default 85 /// maximum distance allowed between end of first wire and start of second for it to be classed as an ROI
        self.minimumDistance = 3  # default 3 /// minimum distance allowed between end of first wire and start of second for it to be classed as an ROI
        self.bboxOffset = 4  # default is 10 /// how many pixels the bounding box extends in the direction of each connecting wire. 0 means the start of wire 1 and end of wire 2 is the bbox.
        # actions / groundSymbolCheck config
        self.groundInspectionAreaHeight = 7   # default is 7 /// half the length of the square inspection area at the end of the wire
        self.groundInspectionAreaWidth = 10  # default is 10 /// half the length of the square inspection area at the end of the wire
        self.groundInspectionThreshold = 0.17    # default is 0.17 /// % of pixels in area to be 'on' pixels to return a found ground symbol

        # Classes / Component config
        self.componentWidth = 25  # default 35 /// how many pixels the bounding box extends at the normal to the direction of each connecting wire

        # io / config
        self.extension = '.tif'
        self.exportPath = 'C:\\PycharmProjects\\MPhil_Project_Final_V0.0.1\\test_results\\'
        self.exportDPI = (300, 300)

        # OCR / config
        # self.langModelSymbols = 'model_project_final'
        # self.langModelLabels = 'eng+grc'
        # self.configLineSymbols = '--psm 13 --dpi 300 -c tessedit_char_whitelist=0123456789@[]+-*{}<>/Aa()\\\ -c load_system_dawg=false -c load_freq_dawg=false'
        # self.configLineLabels = ' -c load_system_dawg=false -c load_freq_dawg=false'
        # self.labelsPSM = '--psm 12'
        self.display = True


config = Config()

class HorizontalLines:
    def __init__(self, x1, y1, x2, y2):
        self.line = x1, y1, x2, y2
        self.length = math.hypot(abs(x2 - x1), abs(y1 - y2))
        self.centre = float(self.length / 2)
        self.start = y1, x1
        self.end = y2, x2
        self.inBox = False
        self.inPair = False


class VerticalLines:
    def __init__(self, x1, y1, x2, y2):
        self.line = x1, y1, x2, y2
        self.length = math.hypot(abs(x2 - x1), abs(y1 - y2))
        self.centre = float(self.length / 2)
        self.start = y1, x1
        self.end = y2, x2
        self.inBox = False
        self.inPair = False


class DiagonalLines:
    def __init__(self, x1, y1, x2, y2):
        self.line = x1, y1, x2, y2
        self.length = math.hypot(abs(x2 - x1), abs(y1 - y2))
        self.centre = float(self.length / 2)
        self.start = y1, x1
        self.end = y2, x2
        self.angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        self.inBox = False
        self.inPair = False


def sortLines(lines):
    """ Sorts lines into horizontal and vertical lists.

    :param lines: list: List of lines returned by houghs probablistic transform.

    :return: list, list: Separate lists of horizontal and vertical lines.
    """
    horizLines = []
    vertLines = []

    for i in range(len(lines)):
        start, end = lines[i]
        x1, y1 = start
        x2, y2 = end
        if (y1 - y2) == 0:
            line = HorizontalLines(x1, y1, x2, y2)
            horizLines.append(line)
        elif (x1 - x2) == 0:
            line = VerticalLines(x1, y1, x2, y2)
            vertLines.append(line)

    return horizLines, vertLines



def wireDetect(border1, border2, wire, threshold=config.threshold):
    """ Detecting whether a wire is present based on a set of parameters

    :param border1: ndarray: Cropped segment of left/top side of the wire.
    :param border2: ndarray: Cropped segment of right/bottom side of the wire.
    :param wire:    ndarray: Cropped segment of the pixels representing the wire.
    :param threshold:   float: % of border pixels that can be filled for border to be counted as empty space.
    :return: bool:  True if wire is found to be present, else False.
    """
    border1Size = np.size(border1)
    border2Size = np.size(border2)
    wireSize = np.size(wire)
    b1Sum = float(np.sum(border1))
    b2Sum = float(np.sum(border2))
    wireSum = float(np.sum(wire))

    if border1Size > 0 and border2Size > 0:
        if b1Sum / border1Size <= threshold and b2Sum / border2Size <= threshold:
            return True
        else:
            return False
    else:
        return False


class WireHoriz:
    def __init__(self, y1, y2, x1, x2, binaryImage):
        self.wire = binaryImage[y1:y2, x1:x2]
        self.length = binaryImage[y1:y2, x1:x2].shape[1]
        self.centre = int(x1 + ((x2 - x1) / 2))
        self.line = y1, y2, x1, x2
        self.start = y1, x1
        self.end = y2, x2
        self.junctionStart = False
        self.junctionEnd = False
        self.componentStart = False
        self.componentEnd = False

    def getBorder(self):
        rows = np.any(self.wire, axis=0)
        columns = np.any(self.wire, axis=1)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(columns)[0][[0, -1]]
        wireBorder = [top, bottom, left, right]
        return wireBorder


class WireVert:
    def __init__(self, y1, y2, x1, x2, binaryImage):
        self.wire = binaryImage[y1:y2, x1:x2]
        self.length = binaryImage[y1:y2, x1:x2].shape[1]
        self.centre = int(y1 + ((y2 - y1) / 2))
        self.line = y1, y2, x1, x2
        self.start = y1, x1
        self.end = y2, x2
        self.junctionStart = False
        self.junctionEnd = False
        self.componentStart = False
        self.componentEnd = False

    def getBorder(self):
        rows = np.any(self.wire, axis=0)
        columns = np.any(self.wire, axis=1)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(columns)[0][[0, -1]]
        wireBorder = [top, bottom, left, right]
        return wireBorder


class WireDiag:
    def __init__(self, y1, y2, x1, x2, binaryImage):
        # For diagonal wires, we need to handle the bounding box properly
        min_y, max_y = min(y1, y2), max(y1, y2)
        min_x, max_x = min(x1, x2), max(x1, x2)
        
        self.wire = binaryImage[min_y:max_y+1, min_x:max_x+1]
        self.length = math.hypot(abs(x2 - x1), abs(y2 - y1))
        self.centre_x = int((x1 + x2) / 2)
        self.centre_y = int((y1 + y2) / 2)
        self.line = y1, y2, x1, x2
        self.start = y1, x1
        self.end = y2, x2
        self.angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        self.junctionStart = False
        self.junctionEnd = False
        self.componentStart = False
        self.componentEnd = False


def wireCheck(Wires, Wire):
    delta = 2
    duplicateWire = False
    wires = []
    for wire in Wires:
        wires.append(wire.line)
    for wire in wires:
        if abs(wire[0] - Wire.line[0]) <= delta and abs(wire[1] - Wire.line[1]) <= delta and abs(
                wire[2] - Wire.line[2]) <= delta and abs(wire[3] - Wire.line[3]) <= delta:
            duplicateWire = True
    return duplicateWire


def sortWiresHough(horizWires, vertWires, diagWires, image):
    """ Bubble sort for found wires. """
    horizWiresSorted = []
    vertWiresSorted = []
    diagWiresSorted = []
    
    # Sort horizontal wires
    for height in range(image.height):
        horizLineSort = []
        for horizWire in horizWires:
            if horizWire.start[0] == height:
                horizLineSort.append(horizWire)
        for _ in horizLineSort:
            for horizLocation in range(len(horizLineSort) - 1):
                if horizLineSort[horizLocation].start[1] > horizLineSort[horizLocation + 1].start[1]:
                    horizLineSort[horizLocation], horizLineSort[horizLocation + 1] = horizLineSort[horizLocation + 1], \
                                                                                     horizLineSort[horizLocation]
        for horizSortedLine in horizLineSort:
            horizWiresSorted.append(horizSortedLine)

    # Sort vertical wires
    for width in range(image.width):
        vertLineSort = []
        for vertWire in vertWires:
            if vertWire.start[1] == width:
                vertLineSort.append(vertWire)
        for __ in vertLineSort:
            for vertLocation in range(len(vertLineSort) - 1):
                if vertLineSort[vertLocation].start[0] > vertLineSort[vertLocation + 1].start[0]:
                    vertLineSort[vertLocation], vertLineSort[vertLocation + 1] = vertLineSort[vertLocation + 1], \
                                                                                 vertLineSort[vertLocation]
        for vertSortedLine in vertLineSort:
            vertWiresSorted.append(vertSortedLine)
    
    # Sort diagonal wires by angle
    diagWiresSorted = sorted(diagWires, key=lambda x: (x.angle, x.start[0], x.start[1]))

    return horizWiresSorted, vertWiresSorted, diagWiresSorted


threshold = 25  # pixels

def is_close_to_endpoints(p, endpoints, threshold=25):
    for e in endpoints:
        if abs(p[0] - e[0]) <= threshold and abs(p[1] - e[1]) <= threshold:
            return True
    return False


def wireScanHough(image, minWireLength=10, borderSize=15):
    """ Scans for wires using Hough's transform - diagonals only if connected to horiz/vert lines """

    HorizWires = []
    VertWires = []
    DiagWires = []

    # --- 1. Detect horizontal + vertical lines ---
    # hv_angles = np.array([0, np.pi/2])
    for loop in range(0, 100):

        angles = np.linspace(0, np.pi / 2, 2)
        lines = probabilistic_hough_line(image.binarySkeleton, threshold=10, line_length=minWireLength,
                                            line_gap=1,
                                            theta=angles)  # finding lines in the image using houghs transform # thresh was 35 for siren
        horizLines, vertLines = sortLines(lines)  # sorting found lines into horizontal and vertical categories

        for line in horizLines:
            left = line.start[1]
            right = line.end[1]

            if line.start[0] - borderSize <= 0:
                top = 0
                bottom = line.start[0] + borderSize
            elif line.start[0] + borderSize >= image.height:
                top = line.start[0] - borderSize
                bottom = image.height
            else:
                top, bottom, = line.start[0] - borderSize, line.start[0] + borderSize

            wire = image.binarySkeleton[line.start[0]:line.start[0] + 1, left:right]
            border1 = image.binarySkeleton[top:line.start[0], left:right]
            border2 = image.binarySkeleton[line.start[0] + 1:bottom, left:right]

            wirePresent = wireDetect(border1, border2, wire)
            if wirePresent:
                wire = WireHoriz(line.start[0], line.start[0], left, right, image.binarySkeleton)
                if not wireCheck(HorizWires, wire):
                    HorizWires.append(wire)

        for line in vertLines:
            bottom = line.start[0]
            top = line.end[0]

            if line.start[1] - borderSize <= 0:
                left = 0
                right = line.start[1] + borderSize
            elif line.start[1] + borderSize >= image.width:
                left = line.start[1] - borderSize
                right = image.width
            else:
                left, right, = line.start[1] - borderSize, line.start[1] + borderSize

            wire = image.binarySkeleton[top:bottom, line.start[1]:line.start[1] + 1]
            border1 = image.binarySkeleton[top:bottom, left:line.start[1]]
            border2 = image.binarySkeleton[top:bottom, line.start[1] + 1:right]

            wirePresent = wireDetect(border1, border2, wire)
            if wirePresent:
                wire = WireVert(top, bottom, line.start[1], line.start[1], image.binarySkeleton)
                if not wireCheck(VertWires, wire):
                    VertWires.append(wire)
        # Process horizontal lines
        for line in horizLines:
            wire = WireHoriz(line.start[0], line.start[0], line.start[1], line.end[1], image.binarySkeleton)
            if not wireCheck(HorizWires, wire):
                HorizWires.append(wire)

        # Process vertical lines
        for line in vertLines:
            wire = WireVert(line.start[0], line.end[0], line.start[1], line.start[1], image.binarySkeleton)
            if not wireCheck(VertWires, wire):
                VertWires.append(wire)

        # Collect endpoints from horiz & vert lines
        endpoints = set()
        for w in HorizWires:
            endpoints.add(w.start)
            endpoints.add(w.end)
        for w in VertWires:
            endpoints.add(w.start)
            endpoints.add(w.end)


        # --- 2. Detect diagonals only at endpoints ---
        diag1 = np.linspace(np.deg2rad(30), np.deg2rad(80), 50)     # Q1
        diag2 = np.linspace(np.deg2rad(100), np.deg2rad(150), 50)   # Q2
        diag3 = np.linspace(np.deg2rad(210), np.deg2rad(260), 50)   # Q3
        diag4 = np.linspace(np.deg2rad(300), np.deg2rad(350), 50)   # Q4

        diag_angles = np.concatenate([diag1, diag2, diag3, diag4])
        diag_lines = probabilistic_hough_line(
            image.binarySkeleton,
            threshold=10,
            line_length=30,
            line_gap=2,
            theta=diag_angles
        )
        for line in diag_lines:
            (x0, y0), (x1, y1) = line
            p1 = (y0, x0)
            p2 = (y1, x1)

            if is_close_to_endpoints(p1, endpoints, threshold) or is_close_to_endpoints(p2, endpoints, threshold):
                wire = WireDiag(y0, y1, x0, x1, image.binarySkeleton)
                if not wireCheck(DiagWires, wire):
                    DiagWires.append(wire)
    print("WIRES", len(HorizWires), len(VertWires), len(DiagWires))
    HorizWires, VertWires, DiagWires = sortWiresHough(HorizWires, VertWires, DiagWires, image)
    return HorizWires, VertWires, DiagWires


def binaryConversion(image):
    """ Converts an image into a boolean binary Image. """
    binaryImage = image > threshold_otsu(image)
    binaryImage = invert(binaryImage)
    return binaryImage


def binarySkeleton(image):
    """ Converts an image into a boolean binarised skeletonised Image. """
    binaryImage = image > threshold_otsu(image)
    binaryImage = invert(binaryImage)
    binarySkeleton = skeletonize(binaryImage)
    return binarySkeleton


class Image:
    def __init__(self, image, path):
        self.name = os.path.splitext(os.path.split(path)[1])[0] if path != "none" else "test_image"
        self.image = image
        self.binaryImage = binaryConversion(self.image)
        self.binarySkeleton = binarySkeleton(self.image)
        self.cleanedImage = []
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.centre = (int(self.width / 2), int(self.height / 2))
        self.size = np.size(image)
        self.path = path


def importImageNew(img):
    """ 
    Converts a given OpenCV image into grayscale and creates Image object.
    """
    # Handle string path
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Could not load image from path: {img}")
    
    # Ensure it's a NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be an OpenCV image (numpy.ndarray) or image path.")
    
    # If grayscale already
    if len(img.shape) == 2:
        return Image(img, "test")

    # Handle color images
    dim3 = img.shape[2]
    if dim3 == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = rgba2rgb(img)
        img = rgb2gray(img)
    elif dim3 == 3:  # RGB/BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = rgb2gray(img)
    else:
        pass  # unexpected, but keep as is

    return Image(img, "none")


def draw_lines_on_image(image_path, horiz_wires, vert_wires, diag_wires):
    """
    Draw detected lines on the original image using different colors for each type.
    
    :param image_path: str: Path to the original image
    :param horiz_wires: list: List of horizontal wires
    :param vert_wires: list: List of vertical wires  
    :param diag_wires: list: List of diagonal wires
    :return: ndarray: Image with lines drawn on it
    """
    # Load original image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path.copy()
    
    if img is None:
        raise ValueError("Could not load image")
    
    # Draw horizontal lines in red
    for wire in horiz_wires:
        y1, y2, x1, x2 = wire.line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
    
    # Draw vertical lines in green
    for wire in vert_wires:
        y1, y2, x1, x2 = wire.line
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
    
    # Draw diagonal lines in blue
    for wire in diag_wires:
        y1, y2, x1, x2 = wire.line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
    
    return img


# Main execution
def main():
    # Load and process image
    # image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\line-detection\output\Type-1\6_1-2) PSSUBN4C10-4\1-2) PSSUBN4C10-4_6_text_redacted.png"
    image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\line-detection\output\Type-1\7_1-2) PSSUBN4C10-4\1-2) PSSUBN4C10-4_7_text_redacted.png"
    # image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\line-detection\output\Type-1\6_44) GSP\44) GSP_6_text_redacted.png"
    # try:
    # image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\junctions_detected.png"

    # Create image object
    image = importImageNew(image_path)
    
    # Detect all types of wires
    HorizWires, VertWires, DiagWires = wireScanHough(image)
    
    # Draw lines on original image
    result_img = draw_lines_on_image(image_path, HorizWires, VertWires, DiagWires)
    
    # Display statistics
    print(f"Detected lines:")
    print(f"  Horizontal: {len(HorizWires)}")
    print(f"  Vertical: {len(VertWires)}")
    print(f"  Diagonal: {len(DiagWires)}")
    print(f"  Total: {len(HorizWires) + len(VertWires) + len(DiagWires)}")
    
    # Show the result
    # cv2.imshow('Detected Lines', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Save the result
    output_path = 'with_lines_diag.png'
    cv2.imwrite(output_path, result_img)
    print(f"Result saved to: {output_path}")
    
    return HorizWires, VertWires, DiagWires, result_img
    
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return None, None, None, None


if __name__ == "__main__":
    HorizWires, VertWires, DiagWires, result_img = main()