import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from skimage import measure, segmentation
from scipy import ndimage as ndi

class medical_image_analyzer:
    def __init__(self):
        self.results = {}

    def select_images(self):
        root = tk.Tk()
        root.withdraw()
        print("Select images")

        image_paths = filedialog.askopenfilenames(
            title="SELECT IMAGES",
            filetypes=[
                ("IMAGES", "*.jpg *.jpeg *.jfif *.png"),
                ("All files", "*.*")
            ]
        )
        return list(image_paths)
    
    def adaptive_threshold(self, gray):
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binary
    
    def morphological_op(self, binary_image):
        # Ensure input is uint8
        if binary_image.dtype != np.uint8:
            binary_image = np.uint8(binary_image)
        
        kernel = np.ones((3,3), np.uint8)
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Ensure output is binary (0 and 255)
        _, closed = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
        return closed
    
    def watershed_segmentation(self, cleaned_image):
        # Ensure the image is binary and of the correct type
        if len(cleaned_image.shape) > 2:
            cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
        
        # Make sure the image is binary (0 and 255)
        _, binary = cv2.threshold(cleaned_image, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate distance transform
        distance_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find sure foreground
        _, sure_fg = cv2.threshold(distance_transform, 0.5 * distance_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find connected components
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all markers so that background is 1, not 0
        markers = markers + 1
        
        # Mark the background region (where binary image is 0) as 0
        markers[binary == 0] = 0
        
        # Apply watershed
        markers = segmentation.watershed(-distance_transform, markers, mask=binary)
        
        return markers, distance_transform

    def feature_extraction(self, markers, original_gray):
        regions = measure.regionprops(markers, intensity_image=original_gray)

        feature_data = []
        for i, region in enumerate(regions):
            if region.area > 50:
                area = region.area
                perimeter = region.perimeter
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                eccentricity = region.eccentricity

                feature_data.append({
                    'Cell_ID': i + 1,
                    'Area': area,
                    'Perimeter': perimeter,
                    'Circularity': circularity,
                    'Eccentricity': eccentricity,
                    'Diagnosis': 'Normal' if circularity > 0.7 and 50 < area < 1000 else 'Abnormal'
                })
        return feature_data

    def template_match(self, gray_image):
        def medical_templates(size, shape='circle'):
            template = np.zeros((size, size), np.uint8)
            center = size // 2

            if shape == 'circle':
                cv2.circle(template, (center, center), size//3, 255, -1)
            elif shape == 'irregular':
                points = np.array([[size//4, size//4], [3*size//4, size//6],
                                   [2*size//3, 3*size//4], [size//3, 2*size//3]])
                cv2.fillPoly(template, [points], 255)
            return template
        
        templates = {
            'Normal_Cell': medical_templates(25, 'circle'),
            'Abnormal_Cell': medical_templates(20, 'irregular')
        }

        detection_results = {}
        for template_name, template in templates.items():
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.6)
            detection_results[template_name] = len(locations[0])

        return detection_results

    def analyze_single_image(self, image_path):
        print(f"INITIALIZING ANALYSIS: {os.path.basename(image_path)}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"LOADING FAILED: {image_path}")
            return None
        
        grayCell = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Image shape: {grayCell.shape}")

        adaptive = self.adaptive_threshold(grayCell)
        print(f"Adaptive threshold shape: {adaptive.shape}")

        cleaned = self.morphological_op(adaptive)
        print(f"Cleaned image shape: {cleaned.shape}")

        markers, dist_transform = self.watershed_segmentation(cleaned)
        print(f"Markers shape: {markers.shape}")
        print(f"Distance transform shape: {dist_transform.shape}")

        features = self.feature_extraction(markers, grayCell)
        template_matching_results = self.template_match(grayCell)
        unique_markers = np.unique(markers)
        cell_count = len(unique_markers) - 1  # excludes background

        result = {
            'image': img,
            'grayCell': grayCell,
            'adaptive': adaptive,
            'cleaned': cleaned,
            'markers': markers,
            'dist_transform': dist_transform,
            'features': features,
            'template_matching_results': template_matching_results,
            'cell_count': cell_count,
            'normal_cells': len([f for f in features if f['Diagnosis'] == 'Normal']),
            'abnormal_cells': len([f for f in features if f['Diagnosis'] == 'Abnormal'])
        }
        return result

    def visualize_results(self, results, image_name):
        result = results[image_name]
        plt.figure(figsize=(25, 15))

        # Row 1: basic processing
        plt.subplot(3, 4, 1)
        plt.imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
        plt.title(f"Original: {os.path.basename(image_name)}", fontweight='bold', fontsize=10)
        plt.axis('off')

        plt.subplot(3, 4, 2)
        plt.imshow(result['adaptive'], cmap='gray')
        plt.title("Adaptive Thresholding", fontweight='bold', fontsize=10)
        plt.axis('off')

        plt.subplot(3, 4, 3)
        plt.imshow(result['cleaned'], cmap='gray')
        plt.title("Morphological cleaning", fontweight='bold', fontsize=10)
        plt.axis('off')

        plt.subplot(3, 4, 4)
        plt.imshow(result['dist_transform'], cmap='hot')
        plt.title("Distance transform", fontweight='bold', fontsize=10)
        plt.axis('off')

        # Row 2 Advanced analysis.
        plt.subplot(3, 4, 5)
        watershed_viz = result['image'].copy()
        watershed_viz[result['markers'] == -1] = [255, 0, 0]
        plt.imshow(cv2.cvtColor(watershed_viz, cv2.COLOR_BGR2RGB))
        plt.title("Watershed segmentation(Red boundaries)", fontweight='bold', fontsize=12)
        plt.axis('off')

        plt.subplot(3, 4, 6)
        # feature visualization area
        feature_viz = np.zeros_like(result['markers'], dtype=float)
        for region in measure.regionprops(result['markers']):
            if region.area > 50:
                feature_viz[result['markers'] == region.label] = region.area
        plt.imshow(feature_viz, cmap='viridis')
        plt.title("Feature Map: Cell Area", fontweight='bold', fontsize=12)
        plt.colorbar()
        plt.axis('off')

        plt.subplot(3, 4, 7)
        # circularity histogram
        if result['features']:
            circularities = [f['Circularity'] for f in result['features']]
            plt.hist(circularities, bins=15, alpha=0.7, color='green')
            plt.title("Circularity Distribution", fontweight='bold', fontsize=12)
            plt.xlabel("Circularity")
            plt.ylabel("Frequency")
            plt.axvline(0.7, color='red', linestyle='--', label='Normal threshold')
            plt.legend()
        
        plt.subplot(3, 4, 8)
        # template matching results
        template_names = list(result['template_matching_results'].keys())
        template_counts = list(result['template_matching_results'].values())
        plt.bar(template_names, template_counts, color=['green', 'red'])
        plt.title("Template Matching results", fontweight='bold', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel("Detections")

        # Row 3 summary and diagnostics
        plt.subplot(3, 4, 9)
        diagnostic_img = result['image'].copy()
        regions = measure.regionprops(result['markers'])

        for region in regions:
            if region.area > 50:
                y, x = region.centroid
                circularity = (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0
                color = (0, 255, 0) if circularity > 0.7 else (255, 0, 0)
                cv2.circle(diagnostic_img, (int(x), int(y)), 3, color, -1)

        plt.imshow(cv2.cvtColor(diagnostic_img, cv2.COLOR_BGR2RGB))
        plt.title("Diagnostic Overview", fontweight='bold', fontsize=12)
        plt.axis('off')

        plt.subplot(3, 4, 10)
        # statistics
        if result['features']:
            areas = [f['Area'] for f in result['features']]
            plt.hist(areas, bins=15, alpha=0.7, color='blue', edgecolor='black')
            plt.title("Cell Area Distribution", fontweight='bold', fontsize=12)
            plt.xlabel("Area(pixels)")
            plt.ylabel("Frequency")

        plt.subplot(3, 4, 11)
        # scatter plot
        if result['features']:
            areas = [f['Area'] for f in result['features']]
            circularities = [f['Circularity'] for f in result['features']]
            colors = ['green' if f['Diagnosis'] == 'Normal' else 'red' for f in result['features']]
            plt.scatter(areas, circularities, c=colors, alpha=0.6)
            plt.title("Areas Vs Circularity", fontweight='bold', fontsize=12)
            plt.xlabel("Area")
            plt.ylabel("Circularity")

        plt.subplot(3, 4, 12)
        # Report
        normal_count = result['normal_cells']
        abnormal_count = result['abnormal_cells']
        total_cells = normal_count + abnormal_count
        abnormality_rate = (abnormal_count / total_cells * 100) if total_cells > 0 else 0

        plt.text(0.1, 0.9, "IMAGE ANALYSIS REPORT", fontweight='bold', fontsize=14, color='blue')
        plt.text(0.1, 0.7, f"Image: {os.path.basename(image_name)}", fontsize=10)
        plt.text(0.1, 0.6, f"Total Cells: {total_cells}", fontsize=10)
        plt.text(0.1, 0.5, f"Normal Cells: {normal_count}", fontsize=10, color='green')
        plt.text(0.1, 0.4, f"Abnormal Cells: {abnormal_count}", fontsize=10, color='red')
        plt.text(0.1, 0.3, f"Abnormality Rate: {abnormality_rate:.1f}%", fontsize=10,
                 color='red' if abnormality_rate > 10 else 'green', fontweight='bold')
        plt.text(0.1, 0.2, f"Template Detections: {sum(result['template_matching_results'].values())}", fontsize=10)
        plt.text(0.1, 0.1, "ANALYSIS COMPLETE!", fontweight='bold', fontsize=10, color='green')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def generate_report(self, results):
        print("\n" + "-"*70)
        print("IMAGE ANALYSIS REPORT")
        print("-"*70)

        total_images = len(results)
        total_cells = sum(r['cell_count'] for r in results.values())
        total_normal = sum(r['normal_cells'] for r in results.values())
        total_abnormal = sum(r['abnormal_cells'] for r in results.values())
        percentage_abnormality = (total_abnormal / total_cells * 100) if total_cells > 0 else 0

        print("SUMMARY OF FINDINGS")
        print(f"    > Images analyzed: {total_images}")
        print(f"    > Total Cells: {total_cells}")
        print(f"    > Normal Cells: {total_normal}")
        print(f"    > Abnormal Cells: {total_abnormal}")
        print(f"    > Percentage Abnormality: {percentage_abnormality:.1f}%")

        print(f"\n INDIVIDUAL IMAGE RESULTS")
        for image_name, result in results.items():
            abnormality_rate = (result['abnormal_cells'] / result['cell_count'] * 100) if result['cell_count'] > 0 else 0
            status = "PATIENT NEEDS IMMEDIATE ATTENTION" if abnormality_rate > 15 else "PATIENT IS NORMAL"
            print(f"    > {os.path.basename(image_name)}: {result['cell_count']} cells, "
                  f"{abnormality_rate:.1f}% abnormal - {status}")
            
        print("RECOMMENDATION")
        if percentage_abnormality > 20:
            print("ABNORMALLY HIGH!\nPatient recommended for further investigation")
        elif percentage_abnormality > 10:
            print("MODERATE ABNORMALITY\nPatient needs close supervision")
        else:
            print("PATIENT IS NORMAL")
        print("-"*70)

    def complete_analysis(self):
        print("ANALYSIS INITIATED...")

        # selecting multiple images
        image_paths = self.select_images()
        if not image_paths:
            print("Error! No images selected")
            return
        print(f"Selected {len(image_paths)} images for analysis")

        # Analyzing each one of the images
        for i, image_path in enumerate(image_paths):
            print(f"\nAnalyzing..{i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

            result = self.analyze_single_image(image_path)
            if result:
                self.results[image_path] = result
                self.visualize_results(self.results, image_path)
                print(f"{os.path.basename(image_path)} analysis complete")

        # generate final summary
        if self.results:
            self.generate_report(self.results)
            print(f"Analysis complete!! Analyzed {len(self.results)} Images")
        else:
            print("No images were analyzed")

# LAUNCH THE PIPELINE
if __name__ == "__main__":
    analyzer = medical_image_analyzer()
    analyzer.complete_analysis()