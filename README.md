import cv2
import numpy as np
import matplotlib.pyplot as plt
#Load the fruit image
image_path = "6.jfif"
image = cv2.imread(r"C:\Users\LENOVO\Desktop\SEM5\IMAGE AND VIDEO ANALYTICS\apple_fruit.jpg")

#Convert the image from BGR (OpenCV default) to RGB (for display)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Display the original image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()
#Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

#Display the preprocessed image
plt.imshow(blurred_image, cmap='gray')
plt.title('Grayscale Blurred Image')
plt.axis('off')
plt.show()
#Apply binary thresholding to detect defects
_, threshold_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY_INV)

#Display the thresholded image
plt.imshow(threshold_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')
plt.show()
#Find contours of the defects
contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Draw the contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

#Convert the image to RGB for proper display
contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

#Display the image with contours
plt.imshow(contour_image_rgb)
plt.title('Detected Defects')
plt.axis('off')
plt.show()

#Print number of detected defects
print(f"Number of detected defects: {len(contours)}")

#Filter contours based on area (e.g., defects larger than a threshold)
min_defect_area = 100  # Adjust this value based on the image resolution
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_defect_area]

#Draw the filtered contours on the original image
filtered_contour_image = image.copy()
cv2.drawContours(filtered_contour_image, filtered_contours, -1, (255, 0, 0), 2)

#Convert the image to RGB for proper display
filtered_contour_image_rgb = cv2.cvtColor(filtered_contour_image, cv2.COLOR_BGR2RGB)

#Display the image with filtered contours
plt.imshow(filtered_contour_image_rgb)
plt.title('Filtered Defects')
plt.axis('off')
plt.show()

#Print number of significant detected defects
print(f"Number of significant detected defects: {len(filtered_contours)}") #it was defected number 36 

#Filter contours based on area (e.g., defects larger than a threshold)
min_defect_area = 100  # Adjust this value based on the image resolution
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_defect_area]

#Draw the filtered contours on the original image
filtered_contour_image = image.copy()
cv2.drawContours(filtered_contour_image, filtered_contours, -1, (255, 0, 0), 2)

#Convert the image to RGB for proper display
filtered_contour_image_rgb = cv2.cvtColor(filtered_contour_image, cv2.COLOR_BGR2RGB)

#Display the image with filtered contours
plt.imshow(filtered_contour_image_rgb)
plt.title('Filtered Defects')
plt.axis('off')
plt.show()

#Print number of significant detected defects
print(f"Number of significant detected defects: {len(filtered_contours)}") #8 defects
#Draw bounding boxes around the defects
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(filtered_contour_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

#Convert the image to RGB for proper display
filtered_contour_image_rgb = cv2.cvtColor(filtered_contour_image, cv2.COLOR_BGR2RGB)

#Display the image with bounding boxes
plt.imshow(filtered_contour_image_rgb)
plt.title('Defected Areas with Bounding Boxes')
plt.axis('off')
plt.show()


