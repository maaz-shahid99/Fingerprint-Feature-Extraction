import cv2
import fingerprint_feature_extractor


def extract_fingerprint_features(image_path, spurious_thresh=10, invert_image=False, 
                                show_result=True, save_result=True):
    # Read the input image in grayscale
    img = cv2.imread(image_path, 0)
    
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    
    # Extract minutiae features
    features_terminations, features_bifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img, 
        spuriousMinutiaeThresh=spurious_thresh, 
        invertImage=invert_image, 
        showResult=show_result, 
        saveResult=save_result
    )
    
    return features_terminations, features_bifurcations


# extract_fingerprint_features(f"C:\\Users\\golut\\OneDrive\\Documents\\Projects\\Test demo 2\\Fingerprint-Feature-Extraction\\enhanced\\1.jpg")

term, bifur = extract_fingerprint_features(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\1_4.jpg", save_result= True)

print(len(bifur))