import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def analyze_images():
    root = tk.Tk()
    root.withdraw()
    print("Select Images")

    image_paths = filedialog.askopenfilenames(
        title="Select image to analyze",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.jfif")]
    )
    
    if image_paths:
        print(f"Images selected: {len(image_paths)}")
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            original_image = cv2.imread(image_path)
            if original_image is not None:
                print(f"ANALYZING MEDICAL IMAGE: {os.path.basename(image_path)}")

                gray_cell = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
                adaptive_binary = cv2.adaptiveThreshold(
                     gray_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

                kernel = np.ones((2,2), np.uint8)
                cleaned_cells = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel)
                cell_contours, _ = cv2.findContours(cleaned_cells, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cell_count = len([cnt for cnt in cell_contours if cv2.contourArea(cnt) > 20])

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                clahe_enhanced = clahe.apply(gray_cell)

                edges = cv2.Canny(clahe_enhanced, 30, 100)
                edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                edge_count = len([cnt for cnt in edge_contours if cv2.contourArea(cnt) > 25])

                cell_image = original_image.copy()
                cv2.drawContours(cell_image, cell_contours, -1, (0, 255, 0), 2)

                edge_image = original_image.copy()
                cv2.drawContours(edge_image, edge_contours, -1, (255, 0, 0), 2)

                combined_image = original_image.copy()
                cv2.drawContours(combined_image, cell_contours, -1, (0, 255, 0), 2)
                cv2.drawContours(combined_image, edge_contours, -1, (255, 0, 0), 1)

                plt.figure(figsize=(20, 12))

                plt.subplot(2, 3, 1)
                plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                plt.title("Original Image", fontsize=14, fontweight='bold')
                plt.axis('off')

                plt.subplot(2, 3, 2)
                plt.imshow(clahe_enhanced, cmap='gray')
                plt.title("CLAHE enhanced(improved quality)", fontsize=14, fontweight='bold')
                plt.axis('off')

                plt.subplot(2, 3, 3)
                plt.imshow(adaptive_binary, cmap='gray')
                plt.title(f"Adaptive Threshold: {cell_count} cells segmented", fontsize=14, fontweight='bold')
                plt.axis('off')

                plt.subplot(2, 3, 4)
                plt.imshow(edges, cmap='gray')
                plt.title(f"Canny edge detection: {edge_count} Boundaries found", fontsize=14, fontweight='bold')
                plt.axis('off')

                plt.subplot(2, 3, 5)
                plt.imshow(cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB))
                plt.title("Cells detected..", fontsize=14, fontweight='bold')
                plt.axis('off')

                plt.subplot(2, 3, 6)
                plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
                plt.title("Combined analyzed image", fontsize=16, fontweight='bold', color='red')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

             #DIAGNOSIS REPORT
                print("\n" + "-"*50)
                print("IMAGE ANALYSIS REPORT")
                print("-"*50)
                print(f"Image: {os.path.basename(image_path)}")
                print(f"Cells seen: {cell_count}")
                print(f"Boundaries seen: {edge_count}")
                print(f"Image dimensions: {gray_cell.shape}" )
                print("Analysis complete")
                print("-"*50)
            else:
                print(f"Loading {image_path} failed")
    else:
        print("No images selected")        

analyze_images()
