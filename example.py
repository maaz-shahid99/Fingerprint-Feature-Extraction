from Scripts.align_img import process_fingerprint

# process_fingerprint(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\68_1.jpg")

#process_fingerprint(r"C:\Users\golut\OneDrive\Desktop\2_3.jpg")

import fingerprint_feature_extractor
import cv2

img = cv2.imread(r"C:\Users\golut\OneDrive\Documents\Projects\Test demo 2\Fingerprint-Feature-Extraction\enhanced\1.jpg", cv2.IMREAD_GRAYSCALE)
fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=False)