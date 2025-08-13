import numpy as np
import math
from PIL import Image

def find_extreme_left_edge_points(image):
    """
    Find two extreme-left edge points Pa and Pb from the fingerprint image
    """
    height, width = image.shape
    left_edge_points = []
    
    # Scan each row from left to right to find leftmost non-zero pixel
    for y in range(height):
        for x in range(width):
            if image[y, x] > 0:  # Assuming foreground pixels are non-zero
                left_edge_points.append((x, y))
                break
    
    if len(left_edge_points) < 2:
        raise ValueError("Cannot find sufficient edge points for alignment")
    
    # Select Pa as the first point and Pb as the last point
    Pa = left_edge_points[0]
    Pb = left_edge_points[-1]
    
    return Pa, Pb

def calculate_rotation_angle(Pa, Pb):
    """
    Calculate the rotation angle based on the line connecting Pa and Pb
    """
    xa, ya = Pa
    xb, yb = Pb
    
    # Calculate angle with respect to x-axis
    delta_x = xb - xa
    delta_y = yb - ya
    
    # Angle in degrees
    theta_rot = math.degrees(math.atan2(delta_y, delta_x))
    
    return theta_rot

def rotate_image(image, angle, center):
    """
    Rotate image by given angle around center point
    """
    # Convert angle to radians
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    height, width = image.shape
    cx, cy = center
    
    # Create output image
    rotated = np.zeros_like(image)
    
    # Apply rotation transformation
    for y in range(height):
        for x in range(width):
            # Translate to origin
            x_translated = x - cx
            y_translated = y - cy
            
            # Rotate
            x_rotated = x_translated * cos_angle - y_translated * sin_angle
            y_rotated = x_translated * sin_angle + y_translated * cos_angle
            
            # Translate back
            x_final = int(x_rotated + cx)
            y_final = int(y_rotated + cy)
            
            # Check bounds and assign pixel value
            if 0 <= x_final < width and 0 <= y_final < height:
                rotated[y_final, x_final] = image[y, x]
    
    return rotated

def align_fingerprint_image(image):
    """
    Main function to align fingerprint image based on the research paper methodology
    """
    # Step 1: Find extreme-left edge points Pa and Pb
    try:
        Pa, Pb = find_extreme_left_edge_points(image)
        print(f"Pa (xa, ya): {Pa}")
        print(f"Pb (xb, yb): {Pb}")
    except ValueError as e:
        print(f"Error: {e}")
        return image
    
    # Step 2: Calculate angle theta_rot
    theta_rot = calculate_rotation_angle(Pa, Pb)
    print(f"Calculated angle θrot: {theta_rot:.2f}°")
    
    # Step 3: Determine rotation direction and angle based on paper's logic
    if theta_rot > 90:
        # Rotate to right direction
        rotation_angle = theta_rot - 90
        direction = "right"
        # For right rotation, use negative angle
        rotation_angle = -rotation_angle
    else:
        # Rotate to left direction  
        rotation_angle = 90 - theta_rot
        direction = "left"
        # For left rotation, use positive angle
    
    print(f"Rotation: {abs(rotation_angle):.2f}° to the {direction}")
    
    # Step 4: Rotate the image
    height, width = image.shape
    center = (width // 2, height // 2)
    
    aligned_image = rotate_image(image, rotation_angle, center)
    
    return aligned_image

# Example usage function
def process_fingerprint(image_path):
    """
    Complete workflow to process a fingerprint image
    """
    # Load image (assuming PIL/Pillow is available)
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        image_array = np.array(img)
        
        # Threshold image to create binary-like representation for edge detection
        threshold = 128
        image_array = (image_array > threshold).astype(np.uint8) * 255
        
        print("Original image loaded and preprocessed")
        print(f"Image dimensions: {image_array.shape}")
        
        # Apply alignment
        aligned_image = align_fingerprint_image(image_array)
        
        # Save aligned image
        aligned_img = Image.fromarray(aligned_image)
        aligned_img.save('aligned_fingerprint.png')
        print("Aligned image saved as 'aligned_fingerprint.png'")
        
        return aligned_image
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# The implementation is ready to use
print("Fingerprint alignment implementation complete!")
print("\nUsage:")
print("1. aligned_image = align_fingerprint_image(your_image_array)")
print("2. process_fingerprint('path_to_your_fingerprint.png')")
