import Scripts.consistent_zone as cz    
import fingerprint_feature_extractor
import cv2
from Scripts.align_img import process_fingerprint

# process_fingerprint(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\68_1.jpg")

process_fingerprint(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\1.jpg")



img = cv2.imread(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\1.jpg", 0)
term, bif = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=False)



# Process the fingerprint
# result = cz.process_fingerprint_with_minutiae_features(
#     img=img,
#     terminations=term,
#     bifurcations=bif,
#     target_width=300,           # Larger zone
#     target_height=300,
#     segment_height=60,          # Bigger segment
#     use_content_bounds=True,    # Use fingerprint boundaries
#     margin=20                   # Safety margin
# )

result = cz.process_fingerprint_with_minutiae_features_pattern_aware(
    img=img,
    terminations=term,
    bifurcations=bif
)
cz.display_pattern_aware_results(result, show_images=True)
