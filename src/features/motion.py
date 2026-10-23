import numpy as np

def extract_motion_features(curr_spatial, prev_spatial, dt):
    """
    Computes a 6-dimensional motion feature vector.
    curr_spatial: spatial vector [cx, cy, w, h, ...]
    prev_spatial: spatial vector [cx, cy, w, h, ...]
    dt: time difference or frame difference (float)
    """
    if prev_spatial is None or dt <= 0:
        return np.zeros(6, dtype=np.float32)
        
    ccx, ccy, cw, ch = curr_spatial[0:4]
    pcx, pcy, pw, ph = prev_spatial[0:4]
    
    dx = (ccx - pcx) / dt
    dy = (ccy - pcy) / dt
    dw = (cw - pw) / dt
    dh = (ch - ph) / dt
    
    velocity = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)
    
    return np.array([dx, dy, dw, dh, velocity, direction], dtype=np.float32)
