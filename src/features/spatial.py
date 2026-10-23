import numpy as np

def extract_spatial_features(bbox, img_width, img_height):
    """
    Computes a 9-dimensional spatial feature vector for a bounding box.
    bbox: [xmin, ymin, xmax, ymax]
    Returns:
        numpy array of shape (9,)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate dimensions
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    # Coordinates normalized
    xn = xmin / img_width
    yn = ymin / img_height
    xx = xmax / img_width
    yx = ymax / img_height
    
    # Normalized center
    cx = xn + (xx - xn) / 2.0
    cy = yn + (yx - yn) / 2.0
    
    # Normalized width/height
    wn = box_w / img_width
    hn = box_h / img_height
    
    # Normalized area
    area = wn * hn
    
    # Aspect ratio
    aspect = wn / hn if hn > 0 else 1.0
    
    return np.array([cx, cy, wn, hn, area, aspect, xn, yn, xx], dtype=np.float32)
