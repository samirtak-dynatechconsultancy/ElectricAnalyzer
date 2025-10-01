"""
Segmentation Actions
====================

Custom tools and functions for segmenting circuit diagrams

author: C. R. Kelly
email: CK598@cam.ac.uk

"""
import numpy as np
import math

from skimage.feature import peak_local_max, match_template
from skimage.transform import probabilistic_hough_line, rotate
import os

from .classes import HorizontalLines, VerticalLines, WireHoriz, WireVert, Component, WireJunctions, WireDiag
from .io import importTemplate, exportTemplate, getNewImageCopy
from .config import Config


config = Config()


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

    # if b1Sum / border1Size <= threshold and b2Sum / border2Size <= threshold and wireSum / wireSize == 1:
    if border1Size > 0 and border2Size > 0:
        if b1Sum / border1Size <= threshold and b2Sum / border2Size <= threshold:
            return True
        else:
            return False
    else:
        return False


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

threshold = 25  # pixels

def is_close_to_endpoints(p, endpoints, threshold=25):
    for e in endpoints:
        if abs(p[0] - e[0]) <= threshold and abs(p[1] - e[1]) <= threshold:
            return True
    return False


def wireScanHough(image, minWireLength=10, borderSize=15, min_final_length=15):
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
        diag_angles = np.concatenate([
            np.linspace(np.deg2rad(15), np.deg2rad(75), 30),
            np.linspace(np.deg2rad(105), np.deg2rad(165), 30),
            np.linspace(np.deg2rad(195), np.deg2rad(255), 30),
            np.linspace(np.deg2rad(285), np.deg2rad(345), 30)
        ])

        diag_lines = probabilistic_hough_line(
            image.binarySkeleton,
            threshold=20,
            line_length=20,
            line_gap=3,
            theta=diag_angles
        )

        for line in diag_lines:
            (x0, y0), (x1, y1) = line
            p1 = (y0, x0)
            p2 = (y1, x1)
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            angle = np.abs(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            angle_deg = np.rad2deg(angle)
            
            if length > 25 and (30 < angle_deg < 60 or 120 < angle_deg < 150):
                wire = WireDiag(y0, y1, x0, x1, image.binarySkeleton)
                if not wireCheck(DiagWires, wire):
                    DiagWires.append(wire)

            
    DiagWires = merge_overlapping_diagonals(DiagWires)

    # HorizWires = [w for w in HorizWires if w.length >= min_final_length]
    # VertWires = [w for w in VertWires if w.length >= min_final_length]
    # DiagWires = [w for w in DiagWires if w.length >= min_final_length]

    print("WIRES", len(HorizWires), len(VertWires), len(DiagWires))
    HorizWires, VertWires, DiagWires = sortWiresHough(HorizWires, VertWires, DiagWires, image)
    return HorizWires, VertWires, DiagWires


def merge_overlapping_diagonals(DiagWires, angle_tolerance=0.2, distance_threshold=10, gap_threshold=15):
    """Merge diagonal wires that represent the same physical line"""
    if not DiagWires:
        return []
    
    merged = []
    used = [False] * len(DiagWires)
    
    for i, wire1 in enumerate(DiagWires):
        if used[i]:
            continue
        
        # Get angle of wire1 (use the stored angle)
        angle1 = np.deg2rad(wire1.angle)
        
        # Find all wires that overlap with wire1
        overlapping = [wire1]
        used[i] = True
        
        for j, wire2 in enumerate(DiagWires[i+1:], start=i+1):
            if used[j]:
                continue
            
            angle2 = np.deg2rad(wire2.angle)
            
            # Check if angles are similar (accounting for 180° periodicity)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, abs(angle_diff - np.pi), abs(angle_diff + np.pi))
            
            if angle_diff > angle_tolerance:
                continue
            
            # Check if wires are close (perpendicular distance)
            dist1 = point_to_line_distance(wire2.start, wire1.start, wire1.end)
            dist2 = point_to_line_distance(wire2.end, wire1.start, wire1.end)
            
            # Only consider if perpendicular distance is small
            if dist1 > distance_threshold and dist2 > distance_threshold:
                continue
            
            # NOW check if they're actually close along the line (not separated by a gap)
            if wires_have_gap(wire1, wire2, gap_threshold):
                continue
            
            overlapping.append(wire2)
            used[j] = True
        
        # Merge all overlapping wires into one
        if len(overlapping) > 1:
            # Find the extreme points
            all_points = []
            for w in overlapping:
                all_points.append(w.start)  # (y, x)
                all_points.append(w.end)    # (y, x)
            
            # Get the two most distant points
            max_dist = 0
            p1, p2 = all_points[0], all_points[1]
            for i in range(len(all_points)):
                for j in range(i+1, len(all_points)):
                    dist = np.sqrt((all_points[i][0] - all_points[j][0])**2 + 
                                   (all_points[i][1] - all_points[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        p1, p2 = all_points[i], all_points[j]
            
            # p1 and p2 are (y, x) tuples
            merged_wire = WireDiag(p1[0], p2[0], p1[1], p2[1], wire1.wire)
            merged.append(merged_wire)
        else:
            merged.append(wire1)
    
    return merged


def wires_have_gap(wire1, wire2, gap_threshold):
    """Check if two diagonal wires have a gap between them along the line direction"""
    # Get all 4 endpoints
    points = [wire1.start, wire1.end, wire2.start, wire2.end]
    
    # Find min distance between any endpoint of wire1 to any endpoint of wire2
    min_dist = float('inf')
    
    # Distance from wire1.start to both wire2 endpoints
    dist1 = np.sqrt((wire1.start[0] - wire2.start[0])**2 + (wire1.start[1] - wire2.start[1])**2)
    dist2 = np.sqrt((wire1.start[0] - wire2.end[0])**2 + (wire1.start[1] - wire2.end[1])**2)
    min_dist = min(min_dist, dist1, dist2)
    
    # Distance from wire1.end to both wire2 endpoints
    dist3 = np.sqrt((wire1.end[0] - wire2.start[0])**2 + (wire1.end[1] - wire2.start[1])**2)
    dist4 = np.sqrt((wire1.end[0] - wire2.end[0])**2 + (wire1.end[1] - wire2.end[1])**2)
    min_dist = min(min_dist, dist3, dist4)
    
    # If the closest endpoints are farther than gap_threshold, there's a gap
    return min_dist > gap_threshold


def point_to_line_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line segment"""
    # All inputs are (y, x) tuples
    py, px = point
    y1, x1 = line_start
    y2, x2 = line_end
    
    # Line length
    line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_len == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Distance formula (perpendicular distance from point to infinite line)
    distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_len
    return distance

def wireAdd(start, end, HorizWires, VertWires, image):
    """ Manually add a wire, debug function

    :param start: tuple: (x, y) coordinates start of the wire
    :param end: tuple: (x, y) coordinates end of the wire
    :param HorizWires: list: List of horizontal wires returned by the HorizWires class.
    :param VertWires: list: List of vertical wires returned by the Vert class.
    :param image: ndarray: Image of circuit schematic
    """
    y1 = start[0]
    x1 = start[1]
    y2 = end[0]
    x2 = end[1]

    if y1 == y2:
        wire = WireHoriz(y1, y2, x1, x2, image.binarySkeleton)
        HorizWires.append(wire)
    elif x1 == x2:
        wire = WireVert(y1, y2, x1, x2, image.binarySkeleton)
        VertWires.append(wire)
    else:
        print("wire is neither horizontal nor vertical, check input")
        return


def objectDetection(HorizWires, VertWires, maximumDistance=config.maximumDistance,
                    minimumDistance=config.minimumDistance):
    """ Detects if objects are present between found wires.

    :param maximumDistance: int: If the ends of two wires are further apart than this distance then a check will not be performed.
    :param HorizWires: list: List of horizontal wires returned by the HorizWires class.
    :param VertWires: list: List of vertical wires returned by the Vert class.
    :param image: class: Class of circuit diagram image
    :return list: Returns a list of found components
    """
    # Detecting objects between detected wires
    foundComponents = []
    for i in range(len(HorizWires) - 1):
        y1, x1 = HorizWires[i].end
        y2, x2 = HorizWires[i + 1].start
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # euclidean distance
        dist = abs(x2 - x1)  # linear plane distance
        if maximumDistance > dist > minimumDistance and abs(y1 - y2) < 3:
            centroid = int(y2 - (0.5 * (y2 - y1))), int(x2 - (0.5 * (x2 - x1)))
            component = Component(centroid)
            component.Width = int(dist) + config.bboxOffset  # offset paramter for bbox
            foundComponents.append(component)

            currentLocation = len(foundComponents) - 1
            component.id = currentLocation
            component.associatedHWires.append(HorizWires[i])
            component.associatedHWires.append(HorizWires[i + 1])
            HorizWires[i].componentEnd = True
            HorizWires[i + 1].componentStart = True

    for i in range(len(VertWires) - 1):
        y1, x1 = VertWires[i].end
        y2, x2 = VertWires[i + 1].start
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # euclidean distance
        dist = abs(y2 - y1)  # linear plane distance
        if maximumDistance > dist > minimumDistance and abs(x1 - x2) < 3:
            centroid = int(y2 - (0.5 * (y2 - y1))), int(x2 - (0.5 * (x2 - x1)))
            component = Component(centroid)
            component.Height = int(dist) + config.bboxOffset  # offset parameter for bbox, default was 15
            foundComponents.append(component)
            currentLocation = len(foundComponents) - 1
            component.id = currentLocation
            component.isVert = True
            component.associatedVWires.append(VertWires[i])
            component.associatedVWires.append(VertWires[i + 1])
            VertWires[i].componentEnd = True
            VertWires[i + 1].componentStart = True

    return foundComponents


def junctionDetection(HorizWires, VertWires):
    """ Detects which wires intersect to form a wire junction.

    :param HorizWires: list: List of horizontal wires returned by the HorizWires class.
    :param VertWires: list: List of vertical wires returned by the Vert class.
    :return list: Returns a list of found junctions
    """

    maxDistance = 5
    maxDistance = maxDistance ** 2
    foundJunctions = []
    id = 65  # ASCII VALUE
    for horizWire in HorizWires:

        for vertWire in VertWires:

            ###### Corner junctions
            hy, hx = horizWire.start
            vy, vx = vertWire.start
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.start[0], horizWire.start[1] - 1
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionStart = True
                vertWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hy, hx = horizWire.start
            vy, vx = vertWire.end
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.start[0], horizWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionStart = True
                vertWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hy, hx = horizWire.end
            vy, vx = vertWire.start
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.end[0], horizWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionEnd = True
                vertWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hy, hx = horizWire.end
            vy, vx = vertWire.end
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.end[0], horizWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionEnd = True
                vertWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            ###### Tri Junctions east & west
            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hstart[1] - vstart[1])
            if vstart[0] < hstart[0] < vend[0] and distance <= maxDistance:
                centroid = horizWire.start[0], horizWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                horizWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hend[1] - vend[1])
            if vstart[0] < hend[0] < vend[0] and distance <= maxDistance:
                centroid = horizWire.end[0], horizWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                horizWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            ###### Tri Junctions north & south
            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hstart[0] - vstart[0])
            if hstart[1] < vstart[1] < hend[1] and distance <= maxDistance:
                centroid = vertWire.start[0], vertWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                vertWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hend[0] - vend[0])
            if hstart[1] < vend[1] < hend[1] and distance <= maxDistance:
                centroid = vertWire.end[0], vertWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                vertWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            ###### Quad
            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            if vstart[0] < hstart[0] < vend[0] and hstart[1] < vstart[1] < hend[1]:
                centroid = horizWire.start[0], vertWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Quad'
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)
    for junction in foundJunctions:
        junction.id = chr(id)
        id += 1
    return foundJunctions


def templateMatch(template, image):
    """ Uses a template image to find reoccuring instances of said template in the image it was taken .

    :param template: ndarray: cropped segment of original image to be used for template matching.
    :param image:  ndarray: original input image where template was taken from.
    :return image: ndarray: input image with areas with detected matches removed .
    """
    # match template, remove found matches
    result = match_template(image, template)
    pairs = peak_local_max(result, min_distance=1, threshold_abs=0.8)

    return pairs


def groundSymbolCheck(wire, direction, image):
    top, bottom, left, right = 0, 0, 0, 0

    # get length and threshold from config
    height = config.groundInspectionAreaHeight
    width = config.groundInspectionAreaWidth
    threshold = config.groundInspectionThreshold
    emptyBbox = 0, 0, 0, 0
    # find if wire is horizontal or vertical
    if abs(wire.start[0] - wire.end[0]) < 2:
        Horizontal = True
    else:
        Horizontal = False

    # area inspection
    if Horizontal:
        if direction == 'start':
            top, bottom, left, right = wire.start[0] - height, wire.start[0] + height, wire.start[1] - (2 * width), \
                                       wire.start[1]

        elif direction == 'end':
            top, bottom, left, right = wire.end[0] - height, wire.end[0] + height, wire.end[1], wire.end[
                1] + (2 * width)

    elif not Horizontal:
        if direction == 'start':
            top, bottom, left, right = wire.start[0] - (2 * height), wire.start[0], wire.start[1] - width, wire.start[
                1] + width

        elif direction == 'end':
            top, bottom, left, right = wire.end[0], wire.end[0] + (2 * height), wire.end[1] - width, wire.end[
                1] + width

    # checking if area exceeds bounds of image

    if top <= 0 or bottom >= image.shape[0] or left <= 0 or right >= image.shape[1]:
        return False, emptyBbox

    area = image[top:bottom, left:right]
    value = 1 - (float(np.sum(area)) / ((2 * width * 2 * height)))

    if value >= threshold:
        print(value)
        bbox = top, bottom, left, right
        return True, bbox
    else:
        return False, emptyBbox


def groundSymbolDetection(HorizWires, VertWires, image):
    # Delete template image files instead of storing them
    deleteTemplateFiles = True

    activeImage = image.image
    groundSymbols = []
    checkList = []
    templates = []
    templateBoxes = []
    for HorizWire in HorizWires:
        if not HorizWire.componentStart and not HorizWire.junctionStart:
            # add wire to list to show its been checked
            checkList.append(HorizWire)
            # perform check for ground symbol
            check, bbox = groundSymbolCheck(HorizWire, 'start', activeImage)

            if check:
                boxes = []
                top, bottom, left, right = bbox
                height = bottom - top
                width = right - left
                # once ground symbol is found, look for duplicates w/ template matching
                foundTemplates = templateMatch(activeImage[top:bottom, left:right], activeImage)
                # export template for later replacement
                exportTemplate(activeImage[top:bottom, left:right], image.name)
                for template in foundTemplates:
                    # append results to list, crop out templates from existing image
                    bbox = template[0], template[0] + height, template[1], template[1] + width
                    boxes.append(bbox)
                    result = bbox
                    groundSymbols.append(result)
                    activeImage[template[0]:template[0] + height, template[1]:template[1] + width] = 1
                templateBoxes.append(boxes)

        elif not HorizWire.componentEnd and not HorizWire.junctionEnd:
            # add wire to list to show its been checked
            checkList.append(HorizWire)
            # perform check for ground symbol
            check, bbox = groundSymbolCheck(HorizWire, 'end', activeImage)

            if check:
                boxes = []
                top, bottom, left, right = bbox
                height = bottom - top
                width = right - left
                # once ground symbol is found, look for duplicates w/ template matching
                foundTemplates = templateMatch(activeImage[top:bottom, left:right], activeImage)
                # export template for later replacement
                exportTemplate(activeImage[top:bottom, left:right], image.name)
                for template in foundTemplates:
                    # append results to list, crop out templates from existing image
                    bbox = template[0], template[0] + height, template[1], template[1] + width
                    boxes.append(bbox)
                    result = bbox
                    groundSymbols.append(result)
                    activeImage[template[0]:template[0] + height, template[1]:template[1] + width] = 1
                templateBoxes.append(boxes)

    for VertWire in VertWires:
        if not VertWire.componentStart and not VertWire.junctionStart:
            # add wire to list to show its been checked
            checkList.append(VertWire)
            # perform check for ground symbol
            check, bbox = groundSymbolCheck(VertWire, 'start', activeImage)

            if check:
                boxes = []
                top, bottom, left, right = bbox
                height = bottom - top
                width = right - left
                # once ground symbol is found, look for duplicates w/ template matching
                foundTemplates = templateMatch(activeImage[top:bottom, left:right], activeImage)
                # export template for later replacement
                exportTemplate(activeImage[top:bottom, left:right], image.name)
                for template in foundTemplates:
                    # append results to list, crop out templates from existing image
                    bbox = template[0], template[0] + height, template[1], template[1] + width
                    boxes.append(bbox)
                    result = bbox
                    groundSymbols.append(result)
                    activeImage[template[0]:template[0] + height, template[1]:template[1] + width] = 1
                templateBoxes.append(boxes)

        elif not VertWire.componentEnd and not VertWire.junctionEnd:
            # add wire to list to show its been checked
            checkList.append(VertWire)
            # perform check for ground symbol
            check, bbox = groundSymbolCheck(VertWire, 'end', activeImage)

            if check:
                boxes = []
                top, bottom, left, right = bbox
                height = bottom - top
                width = right - left
                # once ground symbol is found, look for duplicates w/ template matching
                foundTemplates = templateMatch(activeImage[top:bottom, left:right], activeImage)
                # export template for later replacement
                exportTemplate(activeImage[top:bottom, left:right], image.name)
                for template in foundTemplates:
                    # append results to list, crop out templates from existing image
                    bbox = template[0], template[0] + height, template[1], template[1] + width
                    boxes.append(bbox)
                    result = bbox
                    groundSymbols.append(result)
                    activeImage[template[0]:template[0] + height, template[1]:template[1] + width] = 1
                templateBoxes.append(boxes)

    # deleting cropped templates
    if deleteTemplateFiles:
        for templateNumber, templateBox in enumerate(templateBoxes):
            filePath = os.path.join(config.exportPath, 'templates', image.name, 'template_' + str(templateNumber) + config.extension)
            os.remove(filePath)

    # re-import original image and map to image.image to reverse template cropping
    filePath = image.path
    image.image = importTemplate(filePath)

    return groundSymbols, templates



def cleanLabels(labels, image):
    '''
    :param image:
    :param labels:
    :return:
    '''
    croppedImage = getNewImageCopy(image)
    strings, boxes = labels
    for box in boxes:
        top, bottom, left, right = box
        croppedImage[top:bottom, left:right] = 1
    image.cleanedImage = croppedImage
    return croppedImage

