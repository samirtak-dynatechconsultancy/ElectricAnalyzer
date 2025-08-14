import os
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import math
from .CircuitSchematicImageInterpreter.io import importImage
from .CircuitSchematicImageInterpreter.actions import wireScanHough
from .CircuitSchematicImageInterpreter.classes import Image
from collections import defaultdict, deque
import pandas as pd
import csv
import string
import numpy as np

JUNCTION_INTERSECTION_THRESHOLD = 15  # Distance threshold for wire-junction intersections
WIRE_CONNECTION_THRESHOLD = 15        # Distance threshold for wire endpoint connections
JUNCTION_PARAM_THRESHOLD_START = 0.8 # Parameter threshold for detecting junction at wire start
JUNCTION_PARAM_THRESHOLD_END = 0.8   # Parameter threshold for detecting junction at wire end
MIN_LENGTH = 35
zoom = 4.0
def load_voc_boxes(voc_path):
    tree = ET.parse(voc_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        xmax = int(bndbox.find("xmax").text)
        ymin = int(bndbox.find("ymin").text)
        ymax = int(bndbox.find("ymax").text)
        name = obj.find("name").text
        boxes.append((xmin, ymin, xmax, ymax, name))
    return boxes

def point_in_box(point, boxes):
    """Check if a point is inside any given bounding box."""
    px, py = point
    for xmin, ymin, xmax, ymax, name in boxes:
        if xmin <= px <= xmax and ymin <= py <= ymax:
            return True
    return False

def detect_junctions(img, boxes=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    th_blur = cv2.medianBlur(th, 11)
    contours, _ = cv2.findContours(th_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    junction_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 25 < area < 300:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circularity = 4 * math.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            if circularity > 0.7:
                point = (int(cx), int(cy))
                
                # Skip junction if inside bounding box
                if boxes and point_in_box(point, boxes):
                    continue
                
                junction_points.append(point)
                cv2.circle(img, point, int(r), (0, 255, 0), -1)
    
    return img, junction_points

def try_easyocr(image_path, overlap=50):
    try:
        import easyocr
    except Exception:
        return None, None
    
    print("[INFO] Using EasyOCR with tiling")
    reader = easyocr.Reader(['en'], gpu=False)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print("[ERROR] Could not read image:", image_path)
        return None, None
    
    h, w = image.shape[:2]

    # Tile size is 1/4 of the image dimensions
    tile_w = max(1, w // 4)
    tile_h = max(1, h // 4)

    boxes = []
    for y in range(0, h, tile_h - overlap):
        for x in range(0, w, tile_w - overlap):
            # Extract tile
            x_end = min(x + tile_w, w)
            y_end = min(y + tile_h, h)
            tile = image[y:y_end, x:x_end]

            # OCR on tile
            results = reader.readtext(tile)

            for bbox, text, conf in results:
                xs = [int(pt[0]) for pt in bbox]
                ys = [int(pt[1]) for pt in bbox]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                # Shift coords to full image space
                boxes.append((x + x1, y + y1, x + x2, y + y2, text, float(conf)))

    return boxes, "easyocr"

def try_pytesseract(image_path):
    try:
        import pytesseract
    except Exception:
        return None, None
    print("[INFO] Using PyTesseract")
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        text = data['text'][i].strip()
        if text == "":
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
        boxes.append((int(x), int(y), int(x + w), int(y + h), text, conf))
    return boxes, "pytesseract"

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import cv2
import numpy as np
def extract_shx_and_text(pdf_name, page_num, output_path=None):
    #TODO: Correct this code for rotations
    """
    Extract SHX and extractable text coordinates from a PDF page.
    
    Returns:
        all_boxes: list of tuples (x1, y1, x2, y2, text, confidence)
        method_name: str
    """
    method_name = "shx_text_extraction"
    all_boxes = []

    # --- SHX part ---
    pdf = PdfReader(open(pdf_name, "rb"))
    total_pages = len(pdf.pages)
    if page_num < 1 or page_num > total_pages:
        raise ValueError(f"Invalid page_num: {page_num}, PDF has {total_pages} pages.")

    page = pdf.pages[page_num - 1]
    original_rotation = page.rotation

    objs = []
    if "/Annots" in page:
        for annot in page["/Annots"]:
            obj = annot.get_object()
            if "AutoCAD SHX Text" in obj.values():
                objs.append(obj)

    # --- Render PDF page to image with zoom factor ---
    doc = fitz.open(pdf_name)
    fitz_page = doc[page_num - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=mat)
    img_cv = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    h, w = img_cv.shape[:2]

    # --- Collect SHX text boxes ---
    for obj in objs:
        if "/Contents" in obj and "/Rect" in obj:
            llx, lly, urx, ury = obj["/Rect"]
            x1 = int(llx * zoom)
            y1 = int((page.mediabox.height - ury) * zoom)
            x2 = int(urx * zoom)
            y2 = int((page.mediabox.height - lly) * zoom)
            text_val = obj["/Contents"]
            if original_rotation == 0:
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            if original_rotation == 270:
                new_x1 = y1
                new_y1 = h - x2
                new_x2 = y2
                new_y2 = h - x1
                
                # Ensure coordinates are in correct order (min, max)
                new_x1, new_x2 = min(new_x1, new_x2), max(new_x1, new_x2)
                new_y1, new_y2 = min(new_y1, new_y2), max(new_y1, new_y2)
            

            all_boxes.append((new_x1, new_y1, new_x2, new_y2, text_val, 1.0))
            cv2.rectangle(img_cv, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 2)
            cv2.putText(img_cv, text_val, (new_x1, new_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- Collect extractable text boxes ---
    text_instances = fitz_page.get_text("dict")["blocks"]
    for block in text_instances:
        if block["type"] == 0:  # text block
            for line in block["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    x0, y0, x1, y1 = [int(c * zoom) for c in (x0, y0, x1, y1)]
                    if original_rotation == 0:
                        new_x1, new_y1, new_x2, new_y2 = x0, y0, x1, y1
                    if original_rotation == 270:
                        new_x1 = y0
                        new_y1 = h - x1
                        new_x2 = y1
                        new_y2 = h - x0
                        
                        # Ensure coordinates are in correct order (min, max)
                        new_x1, new_x2 = min(new_x1, new_x2), max(new_x1, new_x2)
                        new_y1, new_y2 = min(new_y1, new_y2), max(new_y1, new_y2)
                    

                    all_boxes.append((new_x1, new_y1, new_x2, new_y2, span["text"], 1.0))
                    cv2.rectangle(img_cv, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 0), 2)
                    cv2.putText(img_cv, span["text"], (new_x1, new_y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return all_boxes, method_name

def redact_image_by_boxes(image, boxes, fill_color=(255,255,255), shrink=2):
    out = image.copy()
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    for (x1, y1, x2, y2, text, conf) in boxes_sorted:
        cv2.rectangle(out, (x1, y1), (x2, y2), fill_color, thickness=-1)
    return out

def find_wire_endpoint_connections(all_wires, threshold=WIRE_CONNECTION_THRESHOLD):
    """
    Find connections between wire endpoints (wire-to-wire connections)
    Returns list of connections: [(wire1_idx, wire1_end, wire2_idx, wire2_end, distance)]
    where wire1_end and wire2_end are 'start' or 'end'
    threshold: Distance threshold for considering wire endpoints as connected
    """
    connections = []
    
    for i, wire1 in enumerate(all_wires):
        y1_1, y2_1, x1_1, x2_1 = wire1.line
        wire1_start = (x1_1, y1_1)
        wire1_end = (x2_1, y2_1)
        
        length_1 = math.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
        if length_1 < MIN_LENGTH:
            continue  # Skip wires that are too short
        for j, wire2 in enumerate(all_wires):
            if i >= j:  # Avoid duplicate pairs and self-comparison
                continue
                
            y1_2, y2_2, x1_2, x2_2 = wire2.line
            wire2_start = (x1_2, y1_2)
            wire2_end = (x2_2, y2_2)
            
            length_2 = math.sqrt((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)
            if length_2 < MIN_LENGTH or length_1 < MIN_LENGTH:
                continue  # Skip wires that are too short
            else:
                # Check all endpoint combinations
                endpoint_pairs = [
                    (wire1_start, 'start', wire2_start, 'start'),
                    (wire1_start, 'start', wire2_end, 'end'),
                    (wire1_end, 'end', wire2_start, 'start'),
                    (wire1_end, 'end', wire2_end, 'end')
                ]
            
            for (p1, end1, p2, end2) in endpoint_pairs:
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance <= threshold:
                    connections.append((i, end1, j, end2, distance))
    
    return connections

def build_circuit_graph(horiz_wires, vert_wires, junction_points, 
                       junction_threshold=JUNCTION_INTERSECTION_THRESHOLD, 
                       wire_connection_threshold=WIRE_CONNECTION_THRESHOLD):
    """
    Build a graph representation of the circuit where:
    - Nodes are wire endpoints and junctions
    - Edges connect wires to junctions and junctions to wires
    - Also connects wire endpoints that are close to each other
    
    Parameters:
    - junction_threshold: Distance threshold for wire-junction intersections
    - wire_connection_threshold: Distance threshold for wire endpoint connections
    """
    all_wires = list(horiz_wires) + list(vert_wires)
    intersections = find_line_junction_intersections(all_wires, junction_points, junction_threshold)
    wire_connections = find_wire_endpoint_connections(all_wires, wire_connection_threshold)
    
    print(f"Found {len(wire_connections)} wire-to-wire endpoint connections (threshold: {wire_connection_threshold})")
    print(f"Found {len(intersections)} wire-junction intersections (threshold: {junction_threshold})")
    
    # Create graph structure
    graph = defaultdict(list)  # node_id -> [connected_node_ids]
    wire_endpoints = {}  # wire_idx -> {'start': node_id, 'end': node_id}
    junction_nodes = {}  # junction_idx -> node_id
    endpoint_to_node = {}  # (wire_idx, 'start'/'end') -> node_id
    node_counter = 0
    
    # Create junction nodes
    for junction_idx, (jx, jy) in enumerate(junction_points):
        junction_nodes[junction_idx] = node_counter
        node_counter += 1
    
    # First pass: create nodes for wire endpoints, considering wire-to-wire connections
    for wire_idx, wire in enumerate(all_wires):
        y1, y2, x1, x2 = wire.line
        
        # Get intersections for this wire with junctions
        wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
        wire_intersections.sort(key=lambda x: x['param'])
        
        # Determine start node
        start_node = None
        
        # Check if wire starts near a junction
        if wire_intersections and wire_intersections[0]['param'] < JUNCTION_PARAM_THRESHOLD_START:
            start_junction_idx = wire_intersections[0]['junction_idx']
            start_node = junction_nodes[start_junction_idx]
        else:
            # Check if this endpoint is already connected to another wire
            existing_node = None
            for (w1_idx, w1_end, w2_idx, w2_end, dist) in wire_connections:
                if (w1_idx == wire_idx and w1_end == 'start'):
                    # This start point connects to wire w2_idx's w2_end
                    other_endpoint = (w2_idx, w2_end)
                    if other_endpoint in endpoint_to_node:
                        existing_node = endpoint_to_node[other_endpoint]
                        break
                elif (w2_idx == wire_idx and w2_end == 'start'):
                    # This start point connects to wire w1_idx's w1_end
                    other_endpoint = (w1_idx, w1_end)
                    if other_endpoint in endpoint_to_node:
                        existing_node = endpoint_to_node[other_endpoint]
                        break
            
            if existing_node is not None:
                start_node = existing_node
            else:
                start_node = node_counter
                node_counter += 1
        
        endpoint_to_node[(wire_idx, 'start')] = start_node
        
        # Determine end node
        end_node = None
        
        # Check if wire ends near a junction
        if wire_intersections and wire_intersections[-1]['param'] > JUNCTION_PARAM_THRESHOLD_END:
            end_junction_idx = wire_intersections[-1]['junction_idx']
            end_node = junction_nodes[end_junction_idx]
        else:
            # Check if this endpoint is already connected to another wire
            existing_node = None
            for (w1_idx, w1_end, w2_idx, w2_end, dist) in wire_connections:
                if (w1_idx == wire_idx and w1_end == 'end'):
                    # This end point connects to wire w2_idx's w2_end
                    other_endpoint = (w2_idx, w2_end)
                    if other_endpoint in endpoint_to_node:
                        existing_node = endpoint_to_node[other_endpoint]
                        break
                elif (w2_idx == wire_idx and w2_end == 'end'):
                    # This end point connects to wire w1_idx's w1_end
                    other_endpoint = (w1_idx, w1_end)
                    if other_endpoint in endpoint_to_node:
                        existing_node = endpoint_to_node[other_endpoint]
                        break
            
            if existing_node is not None:
                end_node = existing_node
            else:
                end_node = node_counter
                node_counter += 1
        
        endpoint_to_node[(wire_idx, 'end')] = end_node
        
        # Store wire endpoints
        wire_endpoints[wire_idx] = {'start': start_node, 'end': end_node}
        
        # Add edges to graph
        if start_node != end_node:  # Avoid self-loops
            graph[start_node].append(end_node)
            graph[end_node].append(start_node)
    
    return graph, wire_endpoints, junction_nodes, intersections

import colorsys
import math
import random

def generate_unique_colors(num_colors, saturation=0.8, value=0.9, seed=42):
    """
    Generate a list of visually distinct colors using HSV color space.
    
    Args:
        num_colors (int): Number of unique colors to generate
        saturation (float): Color saturation (0.0 to 1.0, higher = more vivid)
        value (float): Color brightness (0.0 to 1.0, higher = brighter)
        seed (int): Random seed for reproducible colors
        
    Returns:
        list: List of BGR tuples for OpenCV
    """
    if seed is not None:
        random.seed(seed)
    
    colors = []
    
    if num_colors <= 12:
        # Use predefined high-contrast colors for small numbers
        predefined_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (255, 20, 147), # Deep Pink
            (50, 205, 50),  # Lime Green
            (255, 69, 0),   # Red Orange
        ]
        return predefined_colors[:num_colors]
    
    # For larger numbers, generate colors using golden ratio method
    golden_ratio_conjugate = 0.618033988749895
    
    # Start with a random hue
    h = random.random()
    
    for i in range(num_colors):
        # Use golden ratio to space hues evenly
        h += golden_ratio_conjugate
        h %= 1.0  # Keep hue in [0, 1] range
        
        # Add some variation to saturation and value to increase distinctness
        s = saturation + random.uniform(-0.1, 0.1)
        v = value + random.uniform(-0.1, 0.1)
        
        # Clamp values
        s = max(0.4, min(1.0, s))
        v = max(0.6, min(1.0, v))
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to BGR for OpenCV (values 0-255)
        bgr_color = (int(b * 255), int(g * 255), int(r * 255))
        colors.append(bgr_color)
    
    return colors

def generate_maximally_distinct_colors(num_colors):
    """
    Generate maximally distinct colors using a more sophisticated algorithm.
    Uses CIELAB color space for better perceptual uniformity.
    """
    if num_colors <= 1:
        return [(255, 0, 0)]  # Default red
    
    colors = []
    
    # Start with corners of RGB cube for maximum contrast
    base_colors = [
        (0, 0, 0),       # Black
        (255, 255, 255), # White  
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
    ]
    
    # Use base colors first
    for i in range(min(num_colors, len(base_colors))):
        colors.append(base_colors[i])
    
    # Generate additional colors if needed
    remaining = num_colors - len(colors)
    if remaining > 0:
        # Use HSV space with maximum spacing
        for i in range(remaining):
            hue = (i * 360 / remaining) % 360
            # Alternate between high and medium saturation/value
            sat = 1.0 if i % 2 == 0 else 0.7
            val = 0.9 if i % 3 == 0 else 0.7
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
            bgr_color = (int(b * 255), int(g * 255), int(r * 255))
            colors.append(bgr_color)
    
    return colors

def assign_wire_colors_by_network(horiz_wires, vert_wires, junction_points, 
                                 junction_threshold=JUNCTION_INTERSECTION_THRESHOLD,
                                 wire_connection_threshold=WIRE_CONNECTION_THRESHOLD):
    """
    Assign colors to wires based on connected networks.
    Wires connected through junctions or wire endpoint connections get the same color.
    """    
    all_wires = list(horiz_wires) + list(vert_wires)
    graph, wire_endpoints, junction_nodes, intersections = build_circuit_graph(
        horiz_wires, vert_wires, junction_points)
    
    # Find connected components (networks) using DFS
    visited_nodes = set()
    networks = []
    
    def dfs(node, current_network):
        if node in visited_nodes:
            return
        visited_nodes.add(node)
        current_network.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, current_network)
    
    # Find all connected networks
    for node in graph:
        if node not in visited_nodes:
            network = []
            dfs(node, network)
            if network:
                networks.append(network)
    
    num_networks = len(networks)
    print(f"Found {num_networks} separate electrical networks (including wire-to-wire connections)")
    unique_colors = generate_maximally_distinct_colors(num_networks)

    # Assign colors to networks
    network_colors = {}
    for i, network in enumerate(networks):
        network_color = unique_colors[i]
        for node in network:
            network_colors[node] = network_color
        print(f"Network {i+1}: {len(network)} nodes, Color: {network_color}")
    
    # Assign colors to wire endpoints
    wire_colors = {}
    for wire_idx, endpoints in wire_endpoints.items():
        start_node = endpoints['start']
        end_node = endpoints['end']
        
        start_color = network_colors.get(start_node, (128, 128, 128))  # Gray default
        end_color = network_colors.get(end_node, (128, 128, 128))
        
        # Get wire intersections for this wire
        wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
        
        # Determine which endpoints should be marked
        mark_start = True
        mark_end = True
        
        # Don't mark endpoints that are at junctions (they're connection points)
        for intersection in wire_intersections:
            if intersection['param'] < JUNCTION_PARAM_THRESHOLD_START:  # Near start
                mark_start = False
            if intersection['param'] > JUNCTION_PARAM_THRESHOLD_END:  # Near end
                mark_end = False
        
        # Also don't mark endpoints that are connected to other wires
        # Check if this wire's endpoints are shared with other wires (same node_id)
        wires_sharing_start = sum(1 for ep in wire_endpoints.values() 
                                if ep['start'] == start_node or ep['end'] == start_node)
        wires_sharing_end = sum(1 for ep in wire_endpoints.values() 
                              if ep['start'] == end_node or ep['end'] == end_node)
        
        if wires_sharing_start > 1:  # Shared with other wires
            mark_start = False
        if wires_sharing_end > 1:  # Shared with other wires
            mark_end = False
        
        wire = all_wires[wire_idx]
        y1, y2, x1, x2 = wire.line
        
        wire_colors[wire_idx] = {
            'start_color': start_color if mark_start else None,
            'end_color': end_color if mark_end else None,
            'start_point': (x1, y1),
            'end_point': (x2, y2),
            'network_color': start_color,  # Use for drawing the line itself
            'start_node': start_node,
            'end_node': end_node
        }
        
    return wire_colors, networks, network_colors

def analyze_circuit_flow_improved(horiz_wires, vert_wires, junction_points, threshold=15, wire_connection_threshold=10):
    """
    Improved circuit flow analysis that:
    1. Groups wires by connected networks (including wire-to-wire connections)
    2. Uses alternating colors for source/destination within each network
    3. Ensures proper flow direction visualization
    """    
    all_wires = list(horiz_wires) + list(vert_wires)
    graph, wire_endpoints, junction_nodes, intersections = build_circuit_graph(
        horiz_wires, vert_wires, junction_points)
    
    # Find connected components and assign base colors
    wire_colors, networks, network_colors = assign_wire_colors_by_network(
        horiz_wires, vert_wires, junction_points, threshold, wire_connection_threshold)
    
    num_networks = len(networks)
    unique_base_colors = generate_unique_colors(num_networks)

    # Now implement flow direction logic within each network
    for network_idx, network in enumerate(networks):
        network_wires = []
        network_junctions = []
        
        # Find wires and junctions in this network
        for wire_idx, endpoints in wire_endpoints.items():
            if endpoints['start'] in network or endpoints['end'] in network:
                network_wires.append(wire_idx)
        
        for junction_idx, junction_node in junction_nodes.items():
            if junction_node in network:
                network_junctions.append(junction_idx)
        
        if len(network_wires) <= 1:
            continue  # Single wire networks don't need flow analysis
        
        # Create alternating colors for this network
        base_color = unique_base_colors[network_idx]  # No modulo - always unique!
        source_color = base_color  # Original unique color for sources
        dest_color = tuple(min(255, max(0, int(c * 1.4))) for c in base_color)
        
        # Find source wires (wires with free endpoints not connected to junctions or other wires)
        source_wires = []
        for wire_idx in network_wires:
            wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
            has_free_start = not any(i['param'] < 0.15 for i in wire_intersections)
            has_free_end = not any(i['param'] > 0.85 for i in wire_intersections)
            
            # Also check if endpoints are connected to other wires
            endpoints = wire_endpoints[wire_idx]
            start_node = endpoints['start']
            end_node = endpoints['end']
            
            wires_sharing_start = sum(1 for ep in wire_endpoints.values() 
                                    if ep['start'] == start_node or ep['end'] == start_node)
            wires_sharing_end = sum(1 for ep in wire_endpoints.values() 
                                  if ep['start'] == end_node or ep['end'] == end_node)
            
            if wires_sharing_start > 1:  # Connected to other wires
                has_free_start = False
            if wires_sharing_end > 1:  # Connected to other wires
                has_free_end = False
            
            if has_free_start or has_free_end:
                source_wires.append(wire_idx)
        
        print(f"Network {network_idx + 1}: {len(source_wires)} source wires out of {len(network_wires)} total wires")
        
        # Assign flow colors
        for wire_idx in network_wires:
            wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
            
            # Determine if this is a source wire
            is_source_wire = wire_idx in source_wires
            
            # Get current color assignment
            current_colors = wire_colors[wire_idx]
            
            if is_source_wire:
                # Source wires: start with source color, end with destination color (if not at junction)
                if current_colors['start_color'] is not None:
                    wire_colors[wire_idx]['start_color'] = source_color
                if current_colors['end_color'] is not None:
                    wire_colors[wire_idx]['end_color'] = dest_color
            else:
                # Branch wires: start with destination color, end with source color (if not at junction)
                if current_colors['start_color'] is not None:
                    wire_colors[wire_idx]['start_color'] = dest_color
                if current_colors['end_color'] is not None:
                    wire_colors[wire_idx]['end_color'] = source_color
    
    return wire_colors, networks, network_colors, junction_nodes

def extend_line_through_junctions(wire, intersections_for_wire):
    """
    Modify line endpoints to continue through junctions rather than stopping at them
    """
    if not intersections_for_wire:
        return wire.line
    
    y1, y2, x1, x2 = wire.line
    
    # Sort intersections by parameter (position along line)
    intersections_for_wire.sort(key=lambda x: x['param'])
    
    # Check if we need to extend the line start or end
    first_intersection = intersections_for_wire[0]
    last_intersection = intersections_for_wire[-1]
    
    # If intersection is very close to start/end, don't mark endpoints there
    modified_x1, modified_y1 = x1, y1
    modified_x2, modified_y2 = x2, y2
    
    # If junction is at the start of the line (param < 0.1), extend line backwards
    if first_intersection['param'] < 0.15:
        # Extend line by 20 pixels in the opposite direction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            unit_dx = dx / length
            unit_dy = dy / length
            modified_x1 = x1 - 20 * unit_dx
            modified_y1 = y1 - 20 * unit_dy
    
    # If junction is at the end of the line (param > 0.9), extend line forwards
    if last_intersection['param'] > 0.85:
        # Extend line by 20 pixels in the same direction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            unit_dx = dx / length
            unit_dy = dy / length
            modified_x2 = x2 + 20 * unit_dx
            modified_y2 = y2 + 20 * unit_dy
    
    return (modified_y1, modified_y2, modified_x1, modified_x2)

import cv2
import numpy as np
from skimage.color import rgba2rgb, rgb2gray

def importImageNew(img):
    """ 
    Converts a given OpenCV image into two copies:
    1) Grayscale version
    2) Binarized skeletonized copy (placeholder in this example).

    :param img: ndarray: OpenCV image (BGR, RGB, or grayscale).
    :return: Image object (same type as before, wrapping grayscale).
    """
    # Ensure it's a NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be an OpenCV image (numpy.ndarray).")
    
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

def line_detection_improved(image_path, draw_on_canvas=None, junction_points=None, enable_network_colors=True):
        """
        Improved line detection with better flow analysis and network coloring
        """
    # try:
        # Import image
        image = importImageNew(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Get wires
        HorizWires, VertWires = wireScanHough(image)
        img = image_path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use provided canvas or create a white canvas
        if draw_on_canvas is None:
            canvas = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255
        else:
            canvas = draw_on_canvas.copy()

        # Perform improved flow analysis
        all_wires = list(HorizWires) + list(VertWires)
        intersections = []
        wire_colors = {}
        networks = []
        
        if junction_points and enable_network_colors:
            intersections = find_line_junction_intersections(all_wires, junction_points)
            wire_colors, networks, network_colors, junction_nodes = analyze_circuit_flow_improved(
                HorizWires, VertWires, junction_points)
            print(f"Found {len(intersections)} line-junction intersections")
            print(f"Identified {len(networks)} electrical networks")

        # --- Stage 1: Identify wires connected to junctions ---
        wires_connected_to_junction = set([i['wire_idx'] for i in intersections])

        # --- Stage 2: Filter short wires not connected to junction ---
        def is_long_enough(wire):
            y1, y2, x1, x2 = wire.line  # unpack from the .line property
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            return length >= MIN_LENGTH


        filtered_HorizWires = []
        for idx, wire in enumerate(HorizWires):
            if idx in wires_connected_to_junction or is_long_enough(wire):
                filtered_HorizWires.append(wire)

        filtered_VertWires = []
        for idx, wire in enumerate(VertWires):
            wire_idx_global = len(HorizWires) + idx
            if wire_idx_global in wires_connected_to_junction or is_long_enough(wire):
                filtered_VertWires.append(wire)

        # --- Drawing function ---
        def draw_wire_improved(wire, wire_idx, canvas):
            wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
            modified_line = extend_line_through_junctions(wire, wire_intersections)
            y1, y2, x1, x2 = modified_line
            pt_start = (int(x1), int(y1))
            pt_end = (int(x2), int(y2))

            if wire_idx in wire_colors and enable_network_colors:
                colors = wire_colors[wire_idx]
                line_color = colors.get('network_color', (0, 0, 0))
            else:
                line_color = (0, 0, 0)

            cv2.line(canvas, pt_start, pt_end, line_color, 2, cv2.LINE_AA)

            if wire_idx in wire_colors and enable_network_colors:
                colors = wire_colors[wire_idx]
                if colors['start_color'] is not None:
                    cv2.circle(canvas, pt_start, 6, colors['start_color'], -1)
                    cv2.circle(canvas, pt_start, 6, (0, 0, 0), 2)
                if colors['end_color'] is not None:
                    cv2.circle(canvas, pt_end, 6, colors['end_color'], -1)
                    cv2.circle(canvas, pt_end, 6, (0, 0, 0), 2)
            else:
                mark_start, mark_end = True, True
                for intersection in wire_intersections:
                    if intersection['param'] < 0.15: mark_start = False
                    if intersection['param'] > 0.85: mark_end = False
                if mark_start:
                    cv2.circle(canvas, pt_start, 6, (0, 0, 255), -1)
                    cv2.circle(canvas, pt_start, 6, (0, 0, 0), 2)
                if mark_end:
                    cv2.circle(canvas, pt_end, 6, (0, 0, 255), -1)
                    cv2.circle(canvas, pt_end, 6, (0, 0, 0), 2)

        # Draw filtered wires
        for idx, wire in enumerate(filtered_HorizWires):
            draw_wire_improved(wire, idx, canvas)

        for idx, wire in enumerate(filtered_VertWires):
            draw_wire_improved(wire, len(HorizWires) + idx, canvas)

        print(f"Detected {len(filtered_HorizWires)} horizontal wires and {len(filtered_VertWires)} vertical wires after filtering")

        return canvas, filtered_HorizWires, filtered_VertWires

    # except Exception as e:
    #     print(f"Error in improved line detection: {e}")
    #     return None, [], []

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate the perpendicular distance from a point to a line segment"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:  # Line segment is actually a point
        return math.sqrt(A * A + B * B)
    
    param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)

def find_line_junction_intersections(wires, junction_points, threshold=JUNCTION_INTERSECTION_THRESHOLD):
    """
    Find intersections between wires and junctions
    Returns a list of (wire_index, junction_index, intersection_point)
    threshold: Distance threshold for considering a wire and junction as intersecting
    """
    intersections = []
    
    for wire_idx, wire in enumerate(wires):
        y1, y2, x1, x2 = wire.line
        
        for junction_idx, (jx, jy) in enumerate(junction_points):
            distance = point_to_line_distance(jx, jy, x1, y1, x2, y2)
            
            if distance <= threshold:
                # Find the closest point on the line to the junction
                A = jx - x1
                B = jy - y1
                C = x2 - x1
                D = y2 - y1
                
                dot = A * C + B * D
                len_sq = C * C + D * D
                
                if len_sq > 0:
                    param = dot / len_sq
                    param = max(0, min(1, param))  # Clamp to line segment
                    
                    closest_x = x1 + param * C
                    closest_y = y1 + param * D
                    
                    intersections.append({
                        'wire_idx': wire_idx,
                        'junction_idx': junction_idx,
                        'junction_point': (jx, jy),
                        'line_point': (closest_x, closest_y),
                        'param': param,
                        'distance': distance
                    })
    
    return intersections

def line_in_box(pt_start, pt_end, boxes):
    x1, y1 = pt_start
    x2, y2 = pt_end
    for xmin, ymin, xmax, ymax, name in boxes:
        if (xmin <= x1 <= xmax and ymin <= y1 <= ymax and
            xmin <= x2 <= xmax and ymin <= y2 <= ymax):
            # Also check for vertical/horizontal wire coverage
            if x1 == x2:  # vertical
                if min(y1, y2) >= ymin and max(y1, y2) <= ymax:
                    return True
            elif y1 == y2:  # horizontal
                if min(x1, x2) >= xmin and max(x1, x2) <= xmax:
                    return True
            else:
                # Rare case: diagonal wires (not expected here)
                return True
    return False

import math

def get_simple_connection_list(connection_data):
    """
    Convert connection data to a simple list format.
    
    Returns:
        list: [
            {
                'color': (B, G, R),
                'start_point': (x, y),
                'end_point': (x, y),
                'length': float
            }, ...
        ]
    """
    simple_connections = []
    
    for color, data in connection_data.items():
        for start_pt, end_pt in data['segments']:
            length = math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
            simple_connections.append({
                'color': color,
                'start_point': start_pt,
                'end_point': end_pt,
                'length': length,
                'network_id': data['network_id']
            })
    
    return simple_connections

def get_network_summary(connection_data):
    """
    Get a summary of all networks and their properties.
    
    Returns:
        list: [
            {
                'network_id': int,
                'color': (B, G, R),
                'wire_count': int,
                'total_length': float,
                'start_points': [(x, y), ...],
                'end_points': [(x, y), ...],
                'all_points': [(x, y), ...]  # All unique points in this network
            }, ...
        ]
    """
    network_summary = []
    
    for color, data in connection_data.items():
        # Get all unique points in this network
        all_points = set()
        for start_pt, end_pt in data['segments']:
            all_points.add(start_pt)
            all_points.add(end_pt)
        
        # Extract just the points from start/end point data
        start_points = [pt['point'] for pt in data['start_points']]
        end_points = [pt['point'] for pt in data['end_points']]
        
        network_summary.append({
            'network_id': data['network_id'],
            'color': color,
            'wire_count': data['wire_count'],
            'total_length': data['total_length'],
            'start_points': start_points,
            'end_points': end_points,
            'all_points': list(all_points),
            'segment_count': len(data['segments'])
        })
    
    # Sort by network_id for consistent ordering
    network_summary.sort(key=lambda x: x['network_id'])
    return network_summary

import math

def extract_wire_connection_data(horiz_wires, vert_wires, junction_points, 
                               wire_colors, intersections, bounding_boxes=None):
    """
    Extract wire connection data with start and end points grouped by unique colors.
    
    Returns:
        dict: Dictionary with color as key and list of connected wire segments as value
        {
            color_tuple: {
                'segments': [(start_point, end_point), ...],
                'network_id': int,
                'total_length': float,
                'wire_count': int
            }
        }
    """
    all_wires = list(horiz_wires) + list(vert_wires)
    connection_data = {}
        
    # Process horizontal wires
    for idx, wire in enumerate(horiz_wires):
        wire_intersections = [i for i in intersections if i['wire_idx'] == idx]
        modified_line = extend_line_through_junctions(wire, wire_intersections)
        y1, y2, x1, x2 = modified_line
        pt_start = (int(x1), int(y1))
        pt_end = (int(x2), int(y2))
        
        # Skip if line is in bounding box
        if line_in_box(pt_start, pt_end, bounding_boxes):
            continue
            
        # Get wire colors
        colors = wire_colors.get(idx, {})
        network_color = colors.get('network_color', (0, 0, 0))
        
        # Calculate length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Add to connection data
        if network_color not in connection_data:
            connection_data[network_color] = {
                'segments': [],
                'network_id': len(connection_data),
                'total_length': 0.0,
                'wire_count': 0,
                'start_points': [],
                'end_points': []
            }
        
        connection_data[network_color]['segments'].append((pt_start, pt_end))
        connection_data[network_color]['total_length'] += length
        connection_data[network_color]['wire_count'] += 1
        
        # Track endpoint colors for flow analysis
        start_color = colors.get('start_color')
        end_color = colors.get('end_color')
        
        if start_color is not None:
            connection_data[network_color]['start_points'].append({
                'point': pt_start,
                'color': start_color,
                'wire_idx': idx,
                'type': 'start'
            })
            
        if end_color is not None:
            connection_data[network_color]['end_points'].append({
                'point': pt_end,
                'color': end_color,
                'wire_idx': idx,
                'type': 'end'
            })
    
    # Process vertical wires
    for idx, wire in enumerate(vert_wires):
        wire_idx = len(horiz_wires) + idx
        wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
        modified_line = extend_line_through_junctions(wire, wire_intersections)
        y1, y2, x1, x2 = modified_line
        pt_start = (int(x1), int(y1))
        pt_end = (int(x2), int(y2))
        
        # Skip if line is in bounding box
        if line_in_box(pt_start, pt_end, bounding_boxes):
            continue
            
        # Get wire colors
        colors = wire_colors.get(wire_idx, {})
        network_color = colors.get('network_color', (0, 0, 0))
        
        # Calculate length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Add to connection data
        if network_color not in connection_data:
            connection_data[network_color] = {
                'segments': [],
                'network_id': len(connection_data),
                'total_length': 0.0,
                'wire_count': 0,
                'start_points': [],
                'end_points': []
            }
        
        connection_data[network_color]['segments'].append((pt_start, pt_end))
        connection_data[network_color]['total_length'] += length
        connection_data[network_color]['wire_count'] += 1
        
        # Track endpoint colors for flow analysis
        start_color = colors.get('start_color')
        end_color = colors.get('end_color')
        
        if start_color is not None:
            connection_data[network_color]['start_points'].append({
                'point': pt_start,
                'color': start_color,
                'wire_idx': wire_idx,
                'type': 'start'
            })
            
        if end_color is not None:
            connection_data[network_color]['end_points'].append({
                'point': pt_end,
                'color': end_color,
                'wire_idx': wire_idx,
                'type': 'end'
            })
    
    return connection_data

def get_network_distinct_points(connection_data, junction_points, threshold=WIRE_CONNECTION_THRESHOLD, junction_threshold=JUNCTION_INTERSECTION_THRESHOLD):
    """
    Extract all distinct points from each network's segments.
    Points within threshold distance are filtered out.
    Junction points and points near junctions are ignored.
    
    Args:
        connection_data: Dictionary with network data
        junction_points: List of junction points [(x, y), ...]
        threshold: Distance threshold - if any 2 points are closer than this, exclude them
        junction_threshold: Distance threshold for ignoring points near junctions
    
    Returns:
        dict: Dictionary with color as key and list of filtered distinct points as value
        {
            color_tuple: [(x, y), (x, y), ...],
            ...
        }
    """
    import math
    
    def distance_between_points(p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def is_near_junction(point, junctions, junction_threshold):
        """Check if a point is within junction_threshold of any junction"""
        for junction in junctions:
            if distance_between_points(point, junction) <= junction_threshold:
                return True
        return False
    
    def filter_points_by_threshold(points, threshold):
        """Filter out points that have any other point within threshold distance"""
        if len(points) <= 1:
            return points
        
        filtered_points = []
        
        for i, point in enumerate(points):
            # Check if this point is too close to any other point
            too_close_to_another = False
            
            for j, other_point in enumerate(points):
                if i != j:  # Don't compare point with itself
                    if distance_between_points(point, other_point) < threshold:
                        too_close_to_another = True
                        break
            
            # Only keep points that are not too close to any other point
            if not too_close_to_another:
                filtered_points.append(point)
        
        return filtered_points
    
    network_points = {}
    
    for color, data in connection_data.items():
        # Collect all points from segments
        all_points = []
        for start_pt, end_pt in data['segments']:
            all_points.append(start_pt)
            all_points.append(end_pt)
        
        # Remove exact duplicates while preserving order
        distinct_points = []
        seen = set()
        for point in all_points:
            if point not in seen:
                seen.add(point)
                distinct_points.append(point)
        
        # Filter out points that are near junctions
        non_junction_points = []
        junction_filtered_count = 0
        for point in distinct_points:
            if not is_near_junction(point, junction_points, junction_threshold):
                non_junction_points.append(point)
            else:
                junction_filtered_count += 1
        
        # Filter out points that are too close to each other
        filtered_points = filter_points_by_threshold(non_junction_points, threshold)
        
        network_points[color] = filtered_points
        
        print(f"Network {color}: {len(distinct_points)} distinct -> {len(non_junction_points)} after junction filtering -> {len(filtered_points)} after threshold filtering")
        if junction_filtered_count > 0:
            print(f"  Filtered out {junction_filtered_count} points near junctions")
    
    return network_points


def find_component(point, boxes):
    """Return component name if point lies in a bounding box."""
    x, y = point
    for (x1, y1, x2, y2, name) in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "No Component"

def is_valid_text(text):
    """
    Validate text based on the specified criteria:
    - Can only contain keyboard ASCII characters (32-126)
    - Must be one of: alphabets, numbers, alphanumeric, or combinations with special chars
    - Cannot be only special characters
    - Must contain at least one alphabet or number
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check if all characters are within keyboard ASCII range (32-126)
    # This includes printable characters: space, punctuation, digits, letters
    if not all(32 <= ord(char) <= 126 for char in text):
        return False
    
    # Define character sets
    alphabets = set(string.ascii_letters)  # a-z, A-Z
    numbers = set(string.digits)           # 0-9
    special_chars = set(string.punctuation + ' ')  # Special characters + space
    
    # Categorize characters in the text
    text_chars = set(text)
    has_alphabets = bool(text_chars & alphabets)
    has_numbers = bool(text_chars & numbers)
    has_special = bool(text_chars & special_chars)
    
    # Check if text contains only special characters (not allowed)
    if has_special and not has_alphabets and not has_numbers:
        return False
    
    # Must contain at least one alphabet or number
    if not has_alphabets and not has_numbers:
        return False
    
    return True

def categorize_text(text):
    """Categorize valid text into one of the allowed types."""
    if not is_valid_text(text):
        return "Invalid"
    
    alphabets = set(string.ascii_letters)
    numbers = set(string.digits)
    special_chars = set(string.punctuation + ' ')
    
    text_chars = set(text)
    has_alphabets = bool(text_chars & alphabets)
    has_numbers = bool(text_chars & numbers)
    has_special = bool(text_chars & special_chars)
    
    if has_alphabets and has_numbers and has_special:
        return "Alphanumeric with special characters"
    elif has_alphabets and has_special and not has_numbers:
        return "Alphabets with special characters"
    elif has_numbers and has_special and not has_alphabets:
        return "Numbers with special characters"
    elif has_alphabets and has_numbers and not has_special:
        return "Alphanumeric"
    elif has_alphabets and not has_numbers and not has_special:
        return "Alphabets only"
    elif has_numbers and not has_alphabets and not has_special:
        return "Numbers only"
    
    return "Unknown category"

# def is_valid_circle_terminal(text):
#     """
#     Check if text is a valid terminal for circle components.
#     Valid terminals start with N or L followed by any number.
#     """
#     if not text:
#         return False
    
#     # Remove whitespace and convert to uppercase for consistency
#     clean_text = text.strip().upper()
    
#     # Check if it starts with N or L followed by any characters
#     pattern = r'^[NL].*'
#     return bool(re.match(pattern, clean_text))

def find_closest_text(component, point, text_boxes):
    """
    Find the closest valid text box center to a given point.
    If the closest text is invalid, automatically move to the next closest valid text.
    For circle components, only N* or L* terminals are valid.
    """
    x, y = point
    
    # Check if this is a circle component
    # if component is None or not isinstance(component, str):
    #     return "No Component"
    if not component:
        component = "No Component"
    is_circle_component = "circle" in component.lower()
    
    # Calculate distances for all text boxes
    distances = []
    for i, (x1, y1, x2, y2, text, conf) in enumerate(text_boxes):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.hypot(cx - x, cy - y)
        distances.append((dist, i, text))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[0])
    
    # Find the first valid text
    closest_valid_text = None
    closest_valid_category = None
    all_checked_texts = []
    
    for dist, idx, text in distances:
        # For circle components, use special validation
        # if is_circle_component:
        #     is_valid = is_valid_circle_terminal(text)
        #     all_checked_texts.append((dist, text, is_valid, "circle_terminal"))
        # else:
        is_valid = is_valid_text(text)
        all_checked_texts.append((dist, text, is_valid, "general"))
        
        if is_valid:
            closest_valid_text = text
            break
    
    return closest_valid_text

from .test_only_lines import draw_connections_from_df
def combined_circuit_analysis_improved(pdf_file, page_no, crop_params=None, enable_network_colors=True, wire_connection_threshold=WIRE_CONNECTION_THRESHOLD, xml=None):
    """
    Improved combined function with better flow analysis
    wire_connection_threshold: Distance threshold for detecting wire endpoint connections
    """
    bounding_boxes = []
    if xml is not None:
        bounding_boxes = load_voc_boxes(xml)
    doc = fitz.open(pdf_file)
    fitz_page = doc[page_no - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=mat)
    img_cv = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


    if pix.n == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)


    # Load original image
    img = img_cv
    
    # Apply cropping if specified
    if crop_params:
        x, y, w, h = crop_params
        cropped_img = img[y:y+h, x:x+w]
    else:
        # Default crop parameters
        x, y, w, h = 60, 57, 3249, 2028
        cropped_img = img[y:y+h, x:x+w]
    # cropped_img = img.copy()
    # print(f"Image cropped to region: x={x}, y={y}, w={w}, h={h}")

    # Step 1: Detect junctions
    print("\n=== STEP 1: Junction Detection ===")
    cropped_with_junctions, junction_points = detect_junctions(cropped_img.copy(), bounding_boxes)
    print(f"Detected {len(junction_points)} solid junctions in cropped region")

    print("\n=== STEP 2: Text Detection and Removal ===")
    boxes, used = extract_shx_and_text(pdf_file, page_no)

    cropped_boxes = []
    for box in boxes:
        x1, y1, x2, y2, text_val, conf = box

        # Offset by crop origin
        new_x1 = max(0, x1 - x)
        new_y1 = max(0, y1 - y)
        new_x2 = min(w, x2 - x)
        new_y2 = min(h, y2 - y)

        # Only keep boxes that are at least partially inside crop
        if new_x1 < new_x2 and new_y1 < new_y2:
            cropped_boxes.append((new_x1, new_y1, new_x2, new_y2, text_val, conf))
    boxes = cropped_boxes

    # print(f"REDACTED::::::::::::::::::::::::: {redacted_path}")
    text_redacted_img = redact_image_by_boxes(cropped_with_junctions, boxes, fill_color=(255,255,255), shrink=2)
    print(f"[INFO] OCR Module Used: {used}")

    # Step 3: Improved line detection
    print("\n=== STEP 3: Improved Line Detection ===")
    # Perform improved line detection
    line_canvas, horiz_wires, vert_wires = line_detection_improved(
        text_redacted_img, junction_points=junction_points, enable_network_colors=enable_network_colors)

    if line_canvas is not None:
        line_canvas = cv2.cvtColor(line_canvas, cv2.COLOR_BGR2RGB)

        # Step 4: Create combined visualization
        print("\n=== STEP 4: Combined Visualization ===")
        combined_canvas = cropped_img.copy()
        
        # Perform network analysis for combined view
        all_wires = list(horiz_wires) + list(vert_wires)
        if enable_network_colors:
            intersections = find_line_junction_intersections(all_wires, junction_points)
            wire_colors, networks, network_colors, junction_nodes = analyze_circuit_flow_improved(
                horiz_wires, vert_wires, junction_points)

            # Draw wires with network colors and flow analysis
            for idx, wire in enumerate(horiz_wires):
                wire_intersections = [i for i in intersections if i['wire_idx'] == idx]
                modified_line = extend_line_through_junctions(wire, wire_intersections)
                y1, y2, x1, x2 = modified_line
                pt_start = (int(x1), int(y1))
                pt_end = (int(x2), int(y2))
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Get wire colors
                colors = wire_colors.get(idx, {})
                line_color = colors.get('network_color', (0, 0, 0))
                if line_in_box(pt_start, pt_end, bounding_boxes):
                    continue
                cv2.line(combined_canvas, pt_start, pt_end, line_color, 2, cv2.LINE_AA)
                # Add endpoint markers
                if colors.get('start_color') is not None:
                    cv2.circle(combined_canvas, pt_start, 6, colors['start_color'], -1)
                    cv2.circle(combined_canvas, pt_start, 6, (0, 0, 0), 2)
                if colors.get('end_color') is not None:
                    cv2.circle(combined_canvas, pt_end, 6, colors['end_color'], -1)
                    cv2.circle(combined_canvas, pt_end, 6, (0, 0, 0), 2)
            
            for idx, wire in enumerate(vert_wires):
                wire_idx = len(horiz_wires) + idx
                wire_intersections = [i for i in intersections if i['wire_idx'] == wire_idx]
                modified_line = extend_line_through_junctions(wire, wire_intersections)
                y1, y2, x1, x2 = modified_line
                pt_start = (int(x1), int(y1))
                pt_end = (int(x2), int(y2))
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


                # Get wire colors
                colors = wire_colors.get(wire_idx, {})
                line_color = colors.get('network_color', (0, 0, 0))
                
                if line_in_box(pt_start, pt_end, bounding_boxes):
                    continue
                cv2.line(combined_canvas, pt_start, pt_end, line_color, 2, cv2.LINE_AA)
                
                # Add endpoint markers
                if colors.get('start_color') is not None:
                    cv2.circle(combined_canvas, pt_start, 6, colors['start_color'], -1)
                    cv2.circle(combined_canvas, pt_start, 6, (0, 0, 0), 2)
                if colors.get('end_color') is not None:
                    cv2.circle(combined_canvas, pt_end, 6, colors['end_color'], -1)
                    cv2.circle(combined_canvas, pt_end, 6, (0, 0, 0), 2)
        
        # Re-highlight junctions on top of lines
        for (cx, cy) in junction_points:
            cv2.circle(combined_canvas, (cx, cy), 8, (0, 255, 0), -1)  # Green junctions
            cv2.circle(combined_canvas, (cx, cy), 8, (0, 0, 0), 2)     # Black border
                
        # Print summary
        print(f"\n=== IMPROVED ANALYSIS SUMMARY ===")
        print(f"Junctions detected: {len(junction_points)}")
        print(f"Horizontal wires: {len(horiz_wires)}")
        print(f"Vertical wires: {len(vert_wires)}")
        if enable_network_colors:
            print(f"Electrical networks: {len(networks)}")
        print(f"Text boxes removed: {len(boxes) if boxes else 0}")
    
        connection_data = extract_wire_connection_data(
            horiz_wires, vert_wires, junction_points, 
            wire_colors, intersections, bounding_boxes
        )
        network_points = get_network_distinct_points(connection_data, junction_points, WIRE_CONNECTION_THRESHOLD, JUNCTION_INTERSECTION_THRESHOLD)

        temp = 1
        op_dct = {}
        records = []

        for color, data in connection_data.items():
            print(f"Network Color: {color}")
            print(f"  Wire Count: {data['wire_count']}")
            print(f"  Total Length: {data['total_length']:.1f}px")
            print(f"  Segments: {len(data['segments'])}")
            for i, (start, end) in enumerate(data['segments']):
                print(f"    Segment {i+1}: {start} -> {end}")
            
            distinct_points = network_points[color]
            records.append({"Wire": f"Wire {temp}", "Point": distinct_points})
            op_dct[temp] = distinct_points
            print(f"  Distinct Points ({len(distinct_points)}): {distinct_points}")
            temp += 1

        # Create DataFrame instead of CSV
        df_wire = pd.DataFrame(records)   
        df_wire.to_csv("wire_connections.csv", index=False)
        component_name_counts = {}
        component_id_map = {}

        def get_unique_component_id(component_name, bounding_box):
            """
            Generate a unique identifier for components with the same name.
            Uses the bounding box as a unique key and assigns incremental suffixes.
            """
            # Use bounding box coordinates as a unique identifier for this specific component instance
            bbox_key = tuple(bounding_box) if isinstance(bounding_box, (list, tuple)) else bounding_box
            
            # If we've already assigned an ID to this specific bounding box, return it
            if bbox_key in component_id_map:
                return component_id_map[bbox_key]
            
            # Count occurrences of this component name
            if component_name not in component_name_counts:
                component_name_counts[component_name] = 0
                unique_id = component_name  # First occurrence doesn't need suffix
            else:
                component_name_counts[component_name] += 1
                unique_id = f"{component_name}_{component_name_counts[component_name]}"
            
            # Store the mapping for this bounding box
            component_id_map[bbox_key] = unique_id
            return unique_id

        def find_component_with_id(point, bounding_boxes):
            """
            Find the closest component and return its unique ID.
            Works with bounding_boxes format: [(x1,y1,x2,y2,name), ...]
            """
            if not bounding_boxes:
                return None
            
            px, py = point
            closest_bbox = None
            min_distance = float('inf')
            
            for bbox in bounding_boxes:
                x1, y1, x2, y2, component_name = bbox
                
                # Calculate distance from point to bounding box
                # Find the closest point on the rectangle to the given point
                closest_x = max(x1, min(px, x2))
                closest_y = max(y1, min(py, y2))
                
                # Calculate Euclidean distance
                distance = ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_bbox = bbox
            
            if closest_bbox:
                x1, y1, x2, y2, component_name = closest_bbox
                # Use the first 4 coordinates as the unique bbox identifier
                bbox_coords = (x1, y1, x2, y2)
                return get_unique_component_id(component_name, bbox_coords)
            
            return None

        # Main processing loop
        records_new = []
        for wire_no, points in op_dct.items():
            if not points:
                continue

            # Sort by x first, then y
            sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
            source_point = sorted_points[0]
            
            # Get component with unique ID
            source_component = find_component_with_id(source_point, bounding_boxes)
            source_terminal = find_closest_text(source_component, source_point, boxes)

            for target_point in sorted_points[1:]:
                dest_component = find_component_with_id(target_point, bounding_boxes)
                dest_terminal = find_closest_text(dest_component, target_point, boxes)

                # Always add the record, even if components are None/empty
                records_new.append((wire_no, 
                            source_component if source_component else "", 
                            source_terminal if source_terminal else "",
                            dest_component if dest_component else "", 
                            dest_terminal if dest_terminal else ""))

        print("RECORDDDD")
        print("+++++++++++++++++++++++++++++++++++")
        print(records_new)
        df_connections = pd.DataFrame(records_new, columns=[
            "wire_no", "source_component", "source_terminal",
            "dest_component", "dest_terminal"
        ])


        print("Step 5: Drawing connections from DataFrame")
        drawn_lines_lst = draw_connections_from_df(
            wire_df=df_wire,
            img=cropped_img,
        )

    return df_wire, df_connections, combined_canvas, cropped_with_junctions, line_canvas, drawn_lines_lst