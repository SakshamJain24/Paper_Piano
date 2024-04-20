# import cv2
# import numpy as np

# def detect_and_label_corners(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Detect corners using Shi-Tomasi corner detection
#     corners = cv2.goodFeaturesToTrack(gray, maxCorners=14, qualityLevel=0.01, minDistance=10)

#     # If no corners are found, return the original image
#     if corners is None:
#         return image
    
#     # Convert corners to integers
#     corners = np.int0(corners)
    
#     # Sort corners by x-coordinate
#     corners = sorted(corners, key=lambda x: x[0][0])
    
#     # Draw circles at the corner positions and label them sequentially
#     labeled_image = image.copy()
#     for i, corner in enumerate(corners):
#         x, y = corner.ravel()
#         cv2.circle(labeled_image, (x, y), 5, (0, 255, 0), -1)
#         cv2.putText(labeled_image, f"Corner {i+1}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     return labeled_image

# def main():
#     # Read the input image
#     image = cv2.imread("C:\\Users\\saksh\\OneDrive\\Pictures\\Screenshots\\test.png")  # Replace 'input_image.jpg' with your image file name
    
#     if image is None:
#         print("Error: Unable to load image.")
#         return

#     # Detect and label corners
#     labeled_image = detect_and_label_corners(image)

#     # Display the image with detected and labeled corners
#     cv2.imshow('Image with Labeled Corners', labeled_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np

def detect_and_label_tiles(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=15)

    # If no corners are found, return the original image
    if corners is None:
        return image
    
    # Convert corners to integers
    corners = np.int0(corners)
    
    # Sort corners by x-coordinate
    corners = sorted(corners, key=lambda x: x[0][0])

    # Initialize tile counter and color shade
    tile_counter = 1
    color_shade = 50
    
    # Create a black image with the same size as the original image
    segmented_image = np.zeros_like(image)

    # Iterate over consecutive corners
    for i in range(0, len(corners)-3, 2):
        pt1 = tuple(corners[i].ravel())
        pt2 = tuple(corners[i+1].ravel())
        pt3 = tuple(corners[i+2].ravel())
        pt4 = tuple(corners[i+3].ravel())
        
        # Fill the region between consecutive corners with different colors
        cv2.fillPoly(segmented_image, [np.array([pt1, pt2, pt4, pt3], dtype=np.int32)], (0, 0, color_shade))
        
        # Label the segmented area with tile number
        mid_point = ((pt1[0] + pt2[0] + pt3[0] + pt4[0]) // 4, (pt1[1] + pt2[1] + pt3[1] + pt4[1]) // 4)
        cv2.putText(segmented_image, f"Tile {tile_counter}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Increment tile counter and color shade
        tile_counter += 1
        color_shade += 50

    return segmented_image

def main():
    # Read the input image
    image = cv2.imread("test.png")  # Replace 'input_image.jpg' with your image file name
    
    if image is None:
        print("Error: Unable to load image.")
        return

    # Detect and label tiles
    labeled_image = detect_and_label_tiles(image)

    # Display the labeled image
    cv2.imshow('Labeled Image', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
