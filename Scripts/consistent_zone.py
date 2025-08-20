from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from itertools import permutations
from math import atan2, degrees, pi
from scipy import ndimage


# ============
# Enhanced Data structures
# ============
@dataclass
class FingerprintPattern:
    pattern_type: str  # 'whorl', 'left_loop', 'right_loop', 'arch'
    core_points: List[Tuple[int, int]]
    delta_points: List[Tuple[int, int]]
    confidence: float

@dataclass
class CropResult:
    cropped_img: np.ndarray
    crop_bounds: Tuple[int, int, int, int]  # (y0,y1,x0,x1)
    partial_flags: Dict[str, bool]
    detected_pattern: FingerprintPattern

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
# Advanced Pattern Detection Functions
# ============
def compute_orientation_field_enhanced(img, block_size=12):
    """
    Enhanced ridge orientation field computation with better accuracy.
    """
    h, w = img.shape
    angles = np.zeros((h // block_size, w // block_size))
    coherence = np.zeros((h // block_size, w // block_size))
    
    # Gaussian smoothing for noise reduction
    smoothed = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 1.0)
    
    # Enhanced gradients using Sobel
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_gx = gx[i:i+block_size, j:j+block_size]
            block_gy = gy[i:i+block_size, j:j+block_size]
            
            # Compute structure tensor
            gxx = np.sum(block_gx * block_gx)
            gyy = np.sum(block_gy * block_gy) 
            gxy = np.sum(block_gx * block_gy)
            
            # Local orientation and coherence
            vx = 2 * gxy
            vy = gxx - gyy
            
            angle = 0.5 * np.arctan2(vx, vy) if vy != 0 else 0
            angles[i // block_size, j // block_size] = angle
            
            # Coherence measure (0=random, 1=highly oriented)
            coherence_val = np.sqrt(vx*vx + vy*vy) / (gxx + gyy + 1e-10)
            coherence[i // block_size, j // block_size] = coherence_val
    
    return angles, coherence

def detect_enhanced_singular_points(orientation_field, coherence, threshold=0.3):
    """
    Enhanced singular point detection with pattern classification.
    """
    h, w = orientation_field.shape
    cores = []
    deltas = []
    poincare_map = np.zeros((h, w))
    
    # Compute Poincare index with enhanced accuracy
    for i in range(1, h-1):
        for j in range(1, w-1):
            if coherence[i, j] < 0.3:  # Skip low coherence regions
                continue
                
            # 8-connected neighborhood angles
            angles = [
                orientation_field[i-1, j-1], orientation_field[i-1, j], orientation_field[i-1, j+1],
                orientation_field[i, j+1], orientation_field[i+1, j+1], orientation_field[i+1, j],
                orientation_field[i+1, j-1], orientation_field[i, j-1]
            ]
            
            # Compute angle differences with proper wrapping
            angle_sum = 0
            for k in range(8):
                diff = angles[(k+1) % 8] - angles[k]
                
                # Handle angle wrapping (-π to π)
                if diff > pi/2:
                    diff -= pi
                elif diff < -pi/2:
                    diff += pi
                    
                angle_sum += diff
            
            # Poincare index
            poincare_index = angle_sum / (2 * pi)
            poincare_map[i, j] = poincare_index
            
            # Classify singular points with tighter thresholds
            if abs(poincare_index - 0.5) < threshold:  # Core
                x_coord = j * 12 + 6  # Convert back to image coordinates
                y_coord = i * 12 + 6
                cores.append((x_coord, y_coord))
                
            elif abs(poincare_index + 0.5) < threshold:  # Delta
                x_coord = j * 12 + 6
                y_coord = i * 12 + 6
                deltas.append((x_coord, y_coord))
    
    return cores, deltas, poincare_map

def analyze_ridge_flow_direction(orientation_field, core_point):
    """
    Analyze ridge flow direction around a core to determine loop direction.
    Returns 'left_loop', 'right_loop', or 'whorl'
    """
    if not core_point:
        return 'unknown'
    
    cx, cy = core_point[0] // 12, core_point[1] // 12  # Convert to field coordinates
    h, w = orientation_field.shape
    
    if cx < 1 or cx >= w-1 or cy < 1 or cy >= h-1:
        return 'unknown'
    
    # Sample angles in a circle around the core
    radius = 3
    angles_around_core = []
    
    for angle in np.linspace(0, 2*pi, 16, endpoint=False):
        sample_x = int(cx + radius * np.cos(angle))
        sample_y = int(cy + radius * np.sin(angle))
        
        if 0 <= sample_x < w and 0 <= sample_y < h:
            ridge_angle = orientation_field[sample_y, sample_x]
            angles_around_core.append(ridge_angle)
    
    if len(angles_around_core) < 8:
        return 'unknown'
    
    # Analyze flow pattern
    # For loops, ridges flow in one general direction
    # For whorls, ridges form circular patterns
    
    angle_variance = np.var(angles_around_core)
    angle_range = np.max(angles_around_core) - np.min(angles_around_core)
    
    if angle_variance > 0.8:  # High variance suggests whorl
        return 'whorl'
    
    # Determine loop direction based on average flow
    avg_angle = np.mean(angles_around_core)
    
    # Convert ridge angle to flow direction
    flow_direction = avg_angle + pi/2  # Ridge perpendicular gives flow direction
    
    # Normalize to [0, 2π]
    flow_direction = flow_direction % (2*pi)
    
    # Classify based on flow direction
    # Left loop: flow generally toward left (π/2 to 3π/2)
    # Right loop: flow generally toward right (3π/2 to π/2)
    
    if pi/4 < flow_direction < 3*pi/4:  # Upward flow → right loop
        return 'right_loop'
    elif 5*pi/4 < flow_direction < 7*pi/4:  # Downward flow → left loop  
        return 'left_loop'
    elif flow_direction > 3*pi/4 and flow_direction < 5*pi/4:  # Leftward flow → left loop
        return 'left_loop'
    else:  # Rightward flow → right loop
        return 'right_loop'

def classify_fingerprint_pattern(img):
    """
    Classify fingerprint pattern as whorl, left loop, right loop, or arch.
    Returns FingerprintPattern object with detailed information.
    """
    print("🔍 Analyzing fingerprint pattern...")
    
    # Compute enhanced orientation field
    orientation_field, coherence = compute_orientation_field_enhanced(img)
    
    # Detect singular points
    cores, deltas, poincare_map = detect_enhanced_singular_points(orientation_field, coherence)
    
    print(f"📊 Found {len(cores)} cores and {len(deltas)} deltas")
    
    # Pattern classification logic
    pattern_type = 'arch'  # Default
    confidence = 0.0
    
    if len(cores) == 0 and len(deltas) == 0:
        pattern_type = 'arch'
        confidence = 0.8
        
    elif len(cores) == 1 and len(deltas) <= 1:
        # Likely a loop - determine direction
        loop_direction = analyze_ridge_flow_direction(orientation_field, cores[0])
        if loop_direction in ['left_loop', 'right_loop']:
            pattern_type = loop_direction
            confidence = 0.8
        else:
            pattern_type = 'left_loop'  # Default assumption
            confidence = 0.6
            
    elif len(cores) >= 2 or (len(cores) == 1 and len(deltas) >= 2):
        # Likely a whorl
        pattern_type = 'whorl'
        confidence = 0.9
        
    else:
        # Complex pattern - analyze further
        if len(cores) > len(deltas):
            pattern_type = 'whorl'
            confidence = 0.7
        else:
            # Assume loop and try to determine direction
            loop_direction = analyze_ridge_flow_direction(orientation_field, cores[0] if cores else None)
            pattern_type = loop_direction if loop_direction != 'unknown' else 'left_loop'
            confidence = 0.6
    
    print(f"🎯 Detected pattern: {pattern_type} (confidence: {confidence:.2f})")
    
    return FingerprintPattern(
        pattern_type=pattern_type,
        core_points=cores,
        delta_points=deltas,
        confidence=confidence
    )

def calculate_optimal_crop_region(pattern: FingerprintPattern, img_shape, target_size=(200, 200)):
    """
    Calculate optimal crop region based on detected pattern type and features.
    """
    h, w = img_shape
    target_w, target_h = target_size
    
    # Determine focus points based on pattern type
    if pattern.pattern_type == 'whorl':
        # For whorls, focus on all cores
        if pattern.core_points:
            focus_points = pattern.core_points
        else:
            focus_points = [(w//2, h//2)]
            
    elif pattern.pattern_type in ['left_loop', 'right_loop']:
        # For loops, focus on core and delta (if available)
        focus_points = []
        if pattern.core_points:
            focus_points.extend(pattern.core_points)
        if pattern.delta_points:
            # For loops, delta might be at edge - be more inclusive
            focus_points.extend(pattern.delta_points)
        
        if not focus_points:
            focus_points = [(w//2, h//2)]
            
    else:  # arch or unknown
        # For arches, focus on center or any detected features
        focus_points = pattern.core_points + pattern.delta_points
        if not focus_points:
            focus_points = [(w//2, h//2)]
    
    # Calculate centroid of focus points
    if focus_points:
        focus_x = int(np.mean([p[0] for p in focus_points]))
        focus_y = int(np.mean([p[1] for p in focus_points]))
    else:
        focus_x, focus_y = w//2, h//2
    
    # Calculate crop bounds
    half_w, half_h = target_w // 2, target_h // 2
    
    left = max(0, focus_x - half_w)
    right = min(w, focus_x + half_w)
    top = max(0, focus_y - half_h)
    bottom = min(h, focus_y + half_h)
    
    # Adjust if we hit boundaries
    if right - left < target_w:
        if left == 0:
            right = min(w, target_w)
        elif right == w:
            left = max(0, w - target_w)
    
    if bottom - top < target_h:
        if top == 0:
            bottom = min(h, target_h)
        elif bottom == h:
            top = max(0, h - target_h)
    
    return top, bottom, left, right, (focus_x, focus_y)


# ============
# Enhanced Consistent Zone Selection
# ============
def select_consistent_zone_pattern_aware(
    img: np.ndarray,
    target_width: int = 200,
    target_height: int = 200,
    margin: int = 10
) -> CropResult:
    """
    Pattern-aware consistent zone selection that detects and crops around
    whorl, left loop, right loop, and delta features.
    """
    bw = ensure_binary(img)
    H, W = bw.shape
    
    # Classify the fingerprint pattern
    pattern = classify_fingerprint_pattern(bw)
    
    # Calculate optimal crop region based on pattern
    top, bottom, left, right, focus_point = calculate_optimal_crop_region(
        pattern, (H, W), (target_width, target_height)
    )
    
    # Apply margin constraints
    top = max(margin, top)
    bottom = min(H - margin, bottom)
    left = max(margin, left)
    right = min(W - margin, right)
    
    # Final bounds clamping
    top, bottom, left, right = clamp_bounds(top, bottom, left, right, H, W)
    
    # Extract the pattern-focused consistent zone
    cropped = bw[top:bottom, left:right]
    
    print(f"🎯 Pattern: {pattern.pattern_type}")
    print(f"📍 Focus point: {focus_point}")
    print(f"📐 Crop region: {cropped.shape}")
    if pattern.core_points:
        print(f"🔴 Cores: {pattern.core_points}")
    if pattern.delta_points:
        print(f"🔵 Deltas: {pattern.delta_points}")
    
    return CropResult(
        cropped_img=cropped,
        crop_bounds=(top, bottom, left, right),
        partial_flags={'top': False, 'bottom': False, 'left': False, 'right': False},
        detected_pattern=pattern
    )


# ============
# Keep all previous utility and processing functions
# ============
def extract_minutiae_coordinates(terminations, bifurcations):
    coordinates = []
    for minutiae in terminations:
        coordinates.append((int(minutiae.locX), int(minutiae.locY)))
    for minutiae in bifurcations:
        coordinates.append((int(minutiae.locX), int(minutiae.locY)))
    return coordinates

def ensure_binary(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    vals = np.unique(img)
    if set(vals.tolist()).issubset({0, 1}):
        return img.copy()
    if set(vals.tolist()).issubset({0, 255}):
        return (img // 255).astype(np.uint8)
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
    u = (p_mid[0] - p_from[0], p_mid[1] - p_from[1])
    v = (p_to[0] - p_mid[0], p_to[1] - p_mid[1])
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    ang = degrees(atan2(cross, dot))
    cw = (-ang) % 360.0
    return cw if cw <= 180.0 else 360.0 - cw

# [Include all the horizontal segment and minutiae sequence functions from previous code]
def select_horizontal_segment_with_max_minutiae(minutiae_xy, cropped_shape, h):
    Hc, Wc = cropped_shape
    if Hc <= 0 or h <= 0:
        return SegmentResult(0, 0, [])
    
    h = min(h, Hc)
    diff = np.zeros(Hc + 1, dtype=int)
    valid_points = []
    
    for (x, y) in minutiae_xy:
        if 0 <= y < Hc:
            diff[max(0, y - h + 1)] += 1
            diff[y + 1] -= 1
            valid_points.append((x, y))
    
    counts = np.cumsum(diff[:-1])
    best_y0 = int(np.argmax(counts))
    best_y1 = min(Hc, best_y0 + h)
    
    pts = [(x, y) for (x, y) in valid_points if best_y0 <= y < best_y1]
    return SegmentResult(y_start=best_y0, y_end=best_y1, points_in_segment=pts)

def nearest_neighbor_order(points):
    n = len(points)
    if n == 0:
        return [], []
    
    P = np.asarray(points, dtype=float)
    nn_dists = np.full(n, np.inf)
    
    for i in range(n):
        diff = P - P[i]
        d2 = np.einsum('ij,ij->i', diff, diff)
        d2[i] = np.inf
        j = int(np.argmin(d2))
        nn_dists[i] = np.sqrt(d2[j])
    
    order = list(np.argsort(nn_dists))
    seen = set()
    order = [i for i in order if (i not in seen and not seen.add(i))]
    
    return order, [points[i] for i in order]

def generate_triplet_angles(points, indices):
    if len(indices) < 3:
        return []
    out = []
    for (i, j, k) in permutations(indices, 3):
        ang = angle_clockwise(points[i], points[j], points[k])
        out.append(MinutiaeAngle(triplet_indices=(i, j, k), angle_deg=ang))
    return out

def swap_first_two(triplets, points):
    Mc = []
    for t in triplets:
        i, j, k = t.triplet_indices
        ang = angle_clockwise(points[j], points[i], points[k])
        Mc.append(MinutiaeAngle(triplet_indices=(j, i, k), angle_deg=ang))
    return Mc

def swap_last_two(triplets, points):
    Md = []
    for t in triplets:
        i, j, k = t.triplet_indices
        ang = angle_clockwise(points[i], points[k], points[j])
        Md.append(MinutiaeAngle(triplet_indices=(i, k, j), angle_deg=ang))
    return Md


# ============
# MAIN: Pattern-Aware Processing Pipeline
# ============
def process_fingerprint_pattern_aware(
    img: np.ndarray,
    minutiae_xy: List[Tuple[int, int]],
    target_width: int = 200,
    target_height: int = 200,
    segment_height: int = 50,
    margin: int = 10,
    max_triplets_points: Optional[int] = 50
) -> PipelineResult:
    """
    Pattern-aware fingerprint processing that detects whorl, left loop, right loop,
    and delta features, then crops around them.
    """
    print(f"🚀 Processing fingerprint with {len(minutiae_xy)} minutiae points (PATTERN-AWARE)")
    
    # 1) Pattern-aware consistent zone selection
    crop = select_consistent_zone_pattern_aware(
        img=img,
        target_width=target_width,
        target_height=target_height,
        margin=margin
    )
    
    # 2) Map minutiae to cropped coordinates
    y0, y1, x0, x1 = crop.crop_bounds
    Hc, Wc = crop.cropped_img.shape
    cropped_pts = []
    
    for (x, y) in minutiae_xy:
        xx, yy = x - x0, y - y0
        if 0 <= xx < Wc and 0 <= yy < Hc:
            cropped_pts.append((int(xx), int(yy)))
    
    print(f"🎯 {len(cropped_pts)} minutiae points mapped to pattern-focused zone")
    
    # 3) Select horizontal segment
    segment = select_horizontal_segment_with_max_minutiae(
        cropped_pts, (Hc, Wc), h=segment_height
    )
    
    print(f"📐 Selected segment: y={segment.y_start}-{segment.y_end}, {len(segment.points_in_segment)} points")
    
    # 4) Generate minutiae sequences
    Ma_idx, Ma_points = nearest_neighbor_order(segment.points_in_segment)
    
    if max_triplets_points is not None and len(Ma_idx) > max_triplets_points:
        Ma_idx = Ma_idx[:max_triplets_points]
        Ma_points = Ma_points[:max_triplets_points]
    
    Mb = generate_triplet_angles(Ma_points, list(range(len(Ma_points))))
    Mc = swap_first_two(Mb, Ma_points)
    Md = swap_last_two(Mb, Ma_points)
    
    print(f"✅ Generated sequences: Ma({len(Ma_points)}), Mb({len(Mb)}), Mc({len(Mc)}), Md({len(Md)})")
    
    return PipelineResult(
        crop=crop,
        segment=segment,
        Ma_indices=Ma_idx,
        Ma_points=Ma_points,
        Mb=Mb,
        Mc=Mc,
        Md=Md
    )

def process_fingerprint_with_minutiae_features_pattern_aware(
    img: np.ndarray,
    terminations,  # List of MinutiaeFeature objects
    bifurcations,  # List of MinutiaeFeature objects  
    target_width: int = 200,
    target_height: int = 200,
    segment_height: int = 50,
    margin: int = 10,
    max_triplets_points: Optional[int] = 50
) -> PipelineResult:
    """
    PLUG-AND-PLAY: Pattern-aware fingerprint processing that automatically detects
    whorl, left loop, right loop, and delta features, then crops optimally.
    
    Usage:
        result = process_fingerprint_with_minutiae_features_pattern_aware(
            img=your_aligned_image,
            terminations=your_terminations,
            bifurcations=your_bifurcations
        )
    """
    minutiae_xy = extract_minutiae_coordinates(terminations, bifurcations)
    
    return process_fingerprint_pattern_aware(
        img=img,
        minutiae_xy=minutiae_xy,
        target_width=target_width,
        target_height=target_height,
        segment_height=segment_height,
        margin=margin,
        max_triplets_points=max_triplets_points
    )


# ============
# Enhanced Visualization
# ============
def show_pattern_analysis_results(crop: CropResult, title: str = "Pattern Analysis"):
    """Display cropped zone with detected pattern features marked."""
    img = (crop.cropped_img * 255).astype(np.uint8)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if hasattr(crop, 'detected_pattern'):
        pattern = crop.detected_pattern
        y0, y1, x0, x1 = crop.crop_bounds
        
        # Mark cores in red
        for (cx, cy) in pattern.core_points:
            rel_x, rel_y = cx - x0, cy - y0
            if 0 <= rel_x < color_img.shape[1] and 0 <= rel_y < color_img.shape[0]:
                cv2.circle(color_img, (rel_x, rel_y), 6, (0, 0, 255), 2)  # Red for core
                cv2.putText(color_img, 'C', (rel_x-3, rel_y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Mark deltas in blue
        for (dx, dy) in pattern.delta_points:
            rel_x, rel_y = dx - x0, dy - y0
            if 0 <= rel_x < color_img.shape[1] and 0 <= rel_y < color_img.shape[0]:
                cv2.circle(color_img, (rel_x, rel_y), 6, (255, 0, 0), 2)  # Blue for delta
                cv2.putText(color_img, 'D', (rel_x-3, rel_y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add pattern type text
        cv2.putText(color_img, f'{pattern.pattern_type.upper()}', 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(color_img, f'Conf: {pattern.confidence:.2f}', 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imshow(title, color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_pattern_aware_results(result: PipelineResult, save_images: bool = True, show_images: bool = False):
    """Enhanced display function for pattern-aware results."""
    if save_images:
        success = cv2.imwrite("pattern_focused_zone.png", (result.crop.cropped_img * 255).astype(np.uint8))
        if success:
            print(f"💾 Pattern-focused zone saved to: pattern_focused_zone.png")
    
    if show_images:
        show_pattern_analysis_results(result.crop, "Pattern-Aware Consistent Zone")
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("🔍 PATTERN-AWARE PROCESSING SUMMARY")
    print("="*60)
    
    if hasattr(result.crop, 'detected_pattern'):
        pattern = result.crop.detected_pattern
        print(f"🎯 Detected Pattern: {pattern.pattern_type.upper()}")
        print(f"📊 Confidence: {pattern.confidence:.2f}")
        print(f"🔴 Cores: {len(pattern.core_points)} at {pattern.core_points}")
        print(f"🔵 Deltas: {len(pattern.delta_points)} at {pattern.delta_points}")
    
    print(f"📐 Cropped zone size: {result.crop.cropped_img.shape}")
    print(f"📍 Crop bounds (y0,y1,x0,x1): {result.crop.crop_bounds}")
    print(f"🎯 Selected segment: y={result.segment.y_start}-{result.segment.y_end}")
    print(f"📊 Minutiae in segment: {len(result.segment.points_in_segment)}")
    print(f"🔗 Sequence lengths: Ma({len(result.Ma_points)}), Mb({len(result.Mb)}), Mc({len(result.Mc)}), Md({len(result.Md)})")
