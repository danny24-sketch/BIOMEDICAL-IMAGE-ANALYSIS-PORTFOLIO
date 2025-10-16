#CLAHE medical grade enhancement
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
print("IMAGE ENHANCEMENT INITIATED!")

dataset_path = r"E:\BIOMEDICAL IMAGE ANALYSIS\WEEK 2 Image_processing\blood_cells_data\BCCD_Dataset-master\BCCD"
images_path = os.path.join(dataset_path, "JPEGImages")
blood_cell_images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

first_image_path = os.path.join(images_path, blood_cell_images[0])
blood_cell = cv2.imread(first_image_path)
gray_cell = cv2.cvtColor(blood_cell, cv2.COLOR_BGR2GRAY)
print(f"Processing: {blood_cell_images[1]}")

#regular histogram equalization
regular_enhnced = cv2.equalizeHist(gray_cell)

#CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_enhanced = clahe.apply(gray_cell)
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(gray_cell, cmap='gray')
plt.title("Original cell")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(regular_enhnced, cmap='gray')
plt.title("Regular histogram eqalization")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(clahe_enhanced, cmap='gray')
plt.title("CLAHE enhancement")
plt.axis('off')

#histograms
plt.subplot(2, 3, 4)
plt.hist(gray_cell.ravel(), 256, [0,256], color='black')
plt.title("Original Histogram")

plt.subplot(2, 3, 5)
plt.hist(regular_enhnced.ravel(), 256, [0,256], color='blue')
plt.title("Regular equalized histogram")

plt.subplot(2, 3, 6)
plt.hist(clahe_enhanced.ravel(), 256, [0,256], color='red')
plt.title("CLAHE histogram")

# plt.tight_layout()
# plt.show()
print("CLAHE ENHANCEMENT COMPLETE!")

#CANNY EDGE DETECTION + CONTOURS
print("Canny Edge Detection and contours initiated!")

#using clahe enhanced image for better edge detction
enhanced_for_edges = clahe_enhanced

edges = cv2.Canny(enhanced_for_edges, threshold1=30, threshold2=100)

contours, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Filtering realistic cell contours.
cell_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 25 and cv2.contourArea(cnt)]
print(f"Detected {len(contours)} individual cell boundaries")

#creating visualization
edge_analysis = blood_cell.copy()
cv2.drawContours(edge_analysis, cell_contours, -1, (0, 255, 0), 2)

#Measuring cell properties
print("Image analysis Report:")
for i, contour in enumerate(cell_contours[:5]): #shows first 5 cells
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    print(f"Cell {i+1}: Area={area: .1f}, Perimeter={perimeter: .1f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(enhanced_for_edges, cmap='gray')
plt.title("CLAHE ENHANCED IMAGE")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny edge detection")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(edge_analysis, cv2.COLOR_BGR2RGB))
plt.title(f"Cell boundary Analysis {len(cell_contours)} cells mapped")
plt.axis('off')

plt.tight_layout()
plt.show()
print("CANNY EDGE DETECTION + CONTOURS COMPLETE")

