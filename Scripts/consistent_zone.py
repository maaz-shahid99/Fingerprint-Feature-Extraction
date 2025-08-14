from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from itertools import permutations
from math import atan2, degrees

# ============
# Data structures
# ============
@dataclass
class CropResult:
    cropped_img: np.ndarray
    crop_bounds: Tuple[int, int, int, int]  # (y0,y1,x0,x1)
    partial_flags: Dict[str, bool]

@dataclass
class SegmentResult:
    y_start: int
    y_end: int
    points_in_segment: List[Tuple[int, int]]

@dataclass
class MinutiaeAngle:
    triplet_indices: Tuple[int, int, int]
    angle_deg: float

@dataclass
class PipelineResult:
    crop: CropResult
    segment: SegmentResult
    Ma_indices: List[int]
    Ma_points: List[Tuple[int, int]]
    Mb: List[MinutiaeAngle]
    Mc: List[MinutiaeAngle]
    Md: List[MinutiaeAngle]

# ============
# Utility functions
# ============

# Add this helper function to extract coordinates from MinutiaeFeature objects
def extract_minutiae_coordinates(terminations, bifurcations):
    """
    Extract (x, y) coordinates from MinutiaeFeature objects.
    Combines both terminations and bifurcations into a single list.
    """
    coordinates = []
    
    # Extract from terminations
    for minutiae in terminations:
        coordinates.append((minutiae.X, minutiae.Y))
    
    # Extract from bifurcations  
    for minutiae in bifurcations:
        coordinates.append((minutiae.X, minutiae.Y))
    
    return coordinates

# Modified process_fingerprint function with automatic coordinate extraction
def process_fingerprint_with_minutiae_features(
    img: np.ndarray,
    terminations,  # List of MinutiaeFeature objects
    bifurcations,  # List of MinutiaeFeature objects  
    Top_est: int, Bot_est: int, Left_est: int, Right_est: int,
    H0: int, W0: int,
    segment_height: int,
    edge_band: int = 2,
    edge_thresh_ratio: float = 0.1,
    max_triplets_points: Optional[int] = 50
) -> PipelineResult:
    """
    Modified pipeline that accepts MinutiaeFeature objects directly.
    """
    # Extract coordinates from MinutiaeFeature objects
    minutiae_xy = extract_minutiae_coordinates(terminations, bifurcations)
    
    # Call the original process_fingerprint function
    return process_fingerprint(
        img=img,
        minutiae_xy=minutiae_xy,
        Top_est=Top_est, Bot_est=Bot_est, Left_est=Left_est, Right_est=Right_est,
        H0=H0, W0=W0,
        segment_height=segment_height,
        edge_band=edge_band,
        edge_thresh_ratio=edge_thresh_ratio,
        max_triplets_points=max_triplets_points
    )

def ensure_binary(img: np.ndarray) -> np.ndarray:
    """Ensure binary image is in {0,1} format."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    vals = np.unique(img)
    if set(vals.tolist()).issubset({0, 1}):
        return img.copy()
    if set(vals.tolist()).issubset({0, 255}):
        return (img // 255).astype(np.uint8)
    
    # If values are not 0,1 or 0,255, threshold at midpoint
    return (img > 127).astype(np.uint8)

def clamp_bounds(y0, y1, x0, x1, H, W):
    y0 = max(0, min(y0, H-1))
    y1 = max(0, min(y1, H))
    x0 = max(0, min(x0, W-1))
    x1 = max(0, min(x1, W))
    if y1 <= y0: y1 = min(H, y0 + 1)
    if x1 <= x0: x1 = min(W, x0 + 1)
    return y0, y1, x0, x1

def angle_clockwise(p_from, p_mid, p_to) -> float:
    """
    Clockwise angle in degrees [0,180] measured at p_mid,
    from vector (p_from->p_mid) to (p_mid->p_to).
    """
    u = (p_mid[0] - p_from[0], p_mid[1] - p_from[1])
    v = (p_to[0] - p_mid[0], p_to[1] - p_mid[1])
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    ang = degrees(atan2(cross, dot))  # CCW
    cw = (-ang) % 360.0
    return cw if cw <= 180.0 else 360.0 - cw

# ============
# Partial-edge detection and consistent zone selection
# ============
def edge_band_counts(bw: np.ndarray, band: int = 1) -> Dict[str, int]:
    """
    Count 1s along edge bands (top, bottom, left, right) of thickness 'band'.
    """
    H, W = bw.shape
    b = max(1, min(band, min(H, W)//2))
    return {
        "top": int(bw[0:b, :].sum()),
        "bottom": int(bw[H-b:H, :].sum()),
        "left": int(bw[:, 0:b].sum()),
        "right": int(bw[:, W-b:W].sum()),
    }

def detect_partial_edges(bw: np.ndarray, band: int = 1, thresh_ratio: float = 0.1) -> Dict[str, bool]:
    """
    Decide if edges are partial by comparing band counts to a ratio
    of the band area. Adjust thresh_ratio as needed (empirical).
    """
    H, W = bw.shape
    b = max(1, min(band, min(H, W)//2))
    counts = edge_band_counts(bw, b)
    # Areas for bands
    area_top_bottom = b * W
    area_left_right = b * H
    return {
        "top": counts["top"] >= int(thresh_ratio * area_top_bottom),
        "bottom": counts["bottom"] >= int(thresh_ratio * area_top_bottom),
        "left": counts["left"] >= int(thresh_ratio * area_left_right),
        "right": counts["right"] >= int(thresh_ratio * area_left_right),
    }

def select_consistent_zone(
    img: np.ndarray,
    Top_est: int, Bot_est: int, Left_est: int, Right_est: int,
    H0: int, W0: int,
    edge_band: int = 2,
    edge_thresh_ratio: float = 0.1
) -> CropResult:
    """
    Implements Cases 1–4 with safer edge detection and clamping.
    Returns cropped binary image and metadata.
    """
    bw = ensure_binary(img)
    H00, W00 = bw.shape
    flags = detect_partial_edges(bw, band=edge_band, thresh_ratio=edge_thresh_ratio)
    
    # Default reductions (Case 1 and Case 4 use these directly)
    top_red, bot_red, left_red, right_red = Top_est, Bot_est, Left_est, Right_est
    
    if any(flags.values()):
        top_bottom = flags["top"] and flags["bottom"]
        left_right = flags["left"] and flags["right"]
        
        if not (top_bottom or left_right):
            # Case 2 or 3: adjust only the partial edges
            if flags["top"]:
                top_red = max(0, H00 - (H0 + Bot_est))
            if flags["bottom"]:
                bot_red = max(0, H00 - (H0 + Top_est))
            if flags["left"]:
                left_red = max(0, W00 - (W0 + Right_est))
            if flags["right"]:
                right_red = max(0, W00 - (W0 + Left_est))
        # Else Case 4: keep defaults
    
    # Compute crop bounds
    y0 = int(top_red)
    y1 = int(H00 - bot_red)
    x0 = int(left_red)
    x1 = int(W00 - right_red)
    y0, y1, x0, x1 = clamp_bounds(y0, y1, x0, x1, H00, W00)
    
    # Fallback: if crop smaller than targets and possible, center a window
    ch, cw = y1 - y0, x1 - x0
    if ch < min(H0, H00) or cw < min(W0, W00):
        cy, cx = H00 // 2, W00 // 2
        hh = min(H0, H00) // 2
        ww = min(W0, W00) // 2
        y0, y1 = max(0, cy - hh), min(H00, cy + hh)
        x0, x1 = max(0, cx - ww), min(W00, cx + ww)
        y0, y1, x0, x1 = clamp_bounds(y0, y1, x0, x1, H00, W00)
    
    cropped = bw[y0:y1, x0:x1]
    return CropResult(cropped_img=cropped, crop_bounds=(y0, y1, x0, x1), partial_flags=flags)

# ============
# Horizontal segment selection (prefix-sum approach)
# ============
def select_horizontal_segment_with_max_minutiae(
    minutiae_xy: List[Tuple[int, int]],
    cropped_shape: Tuple[int, int],
    h: int
) -> SegmentResult:
    """
    Fast selection of horizontal band [y, y+h) maximizing minutiae count.
    Uses a difference array/prefix sum over y to get O(H + N).
    """
    Hc, Wc = cropped_shape
    if Hc <= 0 or h <= 0:
        return SegmentResult(0, 0, [])
    
    h = min(h, Hc)
    diff = np.zeros(Hc + 1, dtype=int)
    valid_points: List[Tuple[int, int]] = []
    
    for (x, y) in minutiae_xy:
        if 0 <= y < Hc:
            y0 = y
            y1 = min(Hc, y + h)  # point contributes to all windows starting <= y <= window_end
            diff[max(0, y0 - h + 1)] += 1  # start of contribution range
            diff[y0 + 1] -= 1               # end+1
            valid_points.append((x, y))
    
    counts = np.cumsum(diff[:-1])  # length Hc
    best_y0 = int(np.argmax(counts))
    best_y1 = min(Hc, best_y0 + h)
    
    # Collect points inside the best window
    pts = [(x, y) for (x, y) in valid_points if best_y0 <= y < best_y1]
    return SegmentResult(y_start=best_y0, y_end=best_y1, points_in_segment=pts)

# ============
# Minutiae sequences (Ma, Mb, Mc, Md)
# ============
def nearest_neighbor_order(points: List[Tuple[int, int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Compute nearest-neighbor distance for each point, then order indices
    by ascending NN distance; de-duplicate indices preserving order.
    """
    n = len(points)
    if n == 0:
        return [], []
    
    P = np.asarray(points, dtype=float)  # (n,2)
    order = []
    nn_dists = np.full(n, np.inf)
    nn_idx = np.full(n, -1, dtype=int)
    
    for i in range(n):
        # Vectorized distance compute
        diff = P - P[i]
        d2 = np.einsum('ij,ij->i', diff, diff)
        d2[i] = np.inf
        j = int(np.argmin(d2))
        nn_idx[i] = j
        nn_dists[i] = np.sqrt(d2[j])
    
    order = list(np.argsort(nn_dists))
    # Dedup in case of ties
    seen = set()
    order = [i for i in order if (i not in seen and not seen.add(i))]
    
    return order, [points[i] for i in order]

def generate_triplet_angles(points: List[Tuple[int, int]], indices: List[int]) -> List[MinutiaeAngle]:
    """
    Generate Mb from Ma-ordered indices using all permutations of 3.
    """
    if len(indices) < 3:
        return []
    
    out: List[MinutiaeAngle] = []
    for (i, j, k) in permutations(indices, 3):
        ang = angle_clockwise(points[i], points[j], points[k])
        out.append(MinutiaeAngle(triplet_indices=(i, j, k), angle_deg=ang))
    
    return out

def swap_first_two(triplets: List[MinutiaeAngle], points: List[Tuple[int, int]]) -> List[MinutiaeAngle]:
    Mc: List[MinutiaeAngle] = []
    for t in triplets:
        i, j, k = t.triplet_indices
        ang = angle_clockwise(points[j], points[i], points[k])  # (j,i,k), angle at i
        Mc.append(MinutiaeAngle(triplet_indices=(j, i, k), angle_deg=ang))
    return Mc

def swap_last_two(triplets: List[MinutiaeAngle], points: List[Tuple[int, int]]) -> List[MinutiaeAngle]:
    Md: List[MinutiaeAngle] = []
    for t in triplets:
        i, j, k = t.triplet_indices
        ang = angle_clockwise(points[i], points[k], points[j])  # (i,k,j), angle at k
        Md.append(MinutiaeAngle(triplet_indices=(i, k, j), angle_deg=ang))
    return Md

# ============
# Public pipeline
# ============
def process_fingerprint(
    img: np.ndarray,
    minutiae_xy: List[Tuple[int, int]],
    Top_est: int, Bot_est: int, Left_est: int, Right_est: int,
    H0: int, W0: int,
    segment_height: int,
    edge_band: int = 2,
    edge_thresh_ratio: float = 0.1,
    max_triplets_points: Optional[int] = 50
) -> PipelineResult:
    """
    Complete pipeline:
      1) Select consistent zone with partial-edge handling.
      2) Pick horizontal segment maximizing minutiae count.
      3) Build Ma ordering and Mb/Mc/Md angle sets.
    max_triplets_points: If provided, limit Ma size to control O(n^3) triplet explosion.
    """
    # 1) Crop to consistent zone
    crop = select_consistent_zone(
        img=img,
        Top_est=Top_est, Bot_est=Bot_est, Left_est=Left_est, Right_est=Right_est,
        H0=H0, W0=W0,
        edge_band=edge_band,
        edge_thresh_ratio=edge_thresh_ratio
    )
    
    # Map minutiae to cropped coordinates and filter to bounds
    y0, y1, x0, x1 = crop.crop_bounds
    Hc, Wc = crop.cropped_img.shape
    cropped_pts: List[Tuple[int, int]] = []
    
    for (x, y) in minutiae_xy:
        xx, yy = x - x0, y - y0
        if 0 <= xx < Wc and 0 <= yy < Hc:
            cropped_pts.append((int(xx), int(yy)))
    
    # 2) Select horizontal segment
    segment = select_horizontal_segment_with_max_minutiae(
        cropped_pts, (Hc, Wc), h=segment_height
    )
    
    # 3) Minutiae sequences
    Ma_idx, Ma_points = nearest_neighbor_order(segment.points_in_segment)
    
    # Optional cap to control cubic triplet cost
    if max_triplets_points is not None and len(Ma_idx) > max_triplets_points:
        Ma_idx = Ma_idx[:max_triplets_points]
        Ma_points = Ma_points[:max_triplets_points]
    
    Mb = generate_triplet_angles(Ma_points, list(range(len(Ma_points))))
    Mc = swap_first_two(Mb, Ma_points)
    Md = swap_last_two(Mb, Ma_points)
    
    return PipelineResult(
        crop=crop,
        segment=segment,
        Ma_indices=Ma_idx,
        Ma_points=Ma_points,
        Mb=Mb,
        Mc=Mc,
        Md=Md
    )

# ============
# Visualization and saving helpers (optional)
# ============
def show_cropped(crop: CropResult, title: str = "Cropped Consistent Zone"):
    img = (crop.cropped_img * 255).astype(np.uint8)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_cropped_with_segment(crop: CropResult, segment: SegmentResult, title: str = "Cropped + Segment"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    img = crop.cropped_img
    ys, ye = segment.y_start, segment.y_end
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    rect = patches.Rectangle((0, ys), img.shape[1], ye - ys,
                             linewidth=1.5, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(title)
    ax.axis("off")
    plt.show()

def save_cropped(crop: CropResult, path: str = "consistent_zone.png") -> bool:
    return cv2.imwrite(path, (crop.cropped_img * 255).astype(np.uint8))
