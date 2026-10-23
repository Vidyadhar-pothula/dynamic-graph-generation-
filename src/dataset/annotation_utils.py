def bbox_to_yolo(bbox, img_width, img_height):
    """
    Converts absolute [xmin, ymin, xmax, ymax] coordinates to YOLO format
    [center_x, center_y, width, height] normalized by image dimensions.
    Coordinates are clamped to be within the image bounds.
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Clamp to boundaries
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    
    # Calculate box coordinates
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    # Ensure positive dimension
    if box_w <= 0:
        box_w = 1
    if box_h <= 0:
        box_h = 1
        
    center_x = xmin + (box_w / 2.0)
    center_y = ymin + (box_h / 2.0)
    
    # Normalize
    norm_cx = center_x / img_width
    norm_cy = center_y / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height
    
    return [norm_cx, norm_cy, norm_w, norm_h]

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    Converts normalized YOLO format [center_x, center_y, width, height]
    coordinates back to absolute [xmin, ymin, xmax, ymax].
    """
    cx, cy, w, h = yolo_coords
    
    # Unnormalize
    center_x = cx * img_width
    center_y = cy * img_height
    box_w = w * img_width
    box_h = h * img_height
    
    xmin = int(center_x - (box_w / 2.0))
    ymin = int(center_y - (box_h / 2.0))
    xmax = int(center_x + (box_w / 2.0))
    ymax = int(center_y + (box_h / 2.0))
    
    # Clamp
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    
    return [xmin, ymin, xmax, ymax]
