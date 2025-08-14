from Scripts.consistent_zone import process_fingerprint_with_minutiae_features as process
import fingerprint_feature_extractor
import cv2
from Scripts.align_img import process_fingerprint

# process_fingerprint(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\68_1.jpg")

process_fingerprint(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\63_5.jpg")



# img = cv2.imread(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\1.jpg", 0)
# term, bif = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=False)



# # Process the fingerprint
# result = process(
#     img=img,
#     terminations=term,
#     bifurcations=bif,
#     Top_est=20,
#     Bot_est=20, 
#     Left_est=15,
#     Right_est=15,
#     H0=200,
#     W0=150,
#     segment_height=50
# )