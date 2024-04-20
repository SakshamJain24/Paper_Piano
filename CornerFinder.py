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

# Global variable to store corner positions
fixed_corners = []
tiles = {}

def detect_and_label_corners(image):
    global fixed_corners
    
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
    
    # Store fixed corner positions
    fixed_corners = [corner.ravel() for corner in corners]
    
    # Draw circles at the corner positions and label them sequentially
    labeled_image = image.copy()
    for i, corner in enumerate(fixed_corners):
        x, y = corner
        cv2.circle(labeled_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(labeled_image, f"Corner {i+1}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return labeled_image, fixed_corners

def assign_tiles_to_corners():
    global fixed_corners
    global tiles
    
    num_corners = len(fixed_corners)
    num_tiles = int(num_corners / 2) - 1

    # Initialize the tiles dictionary
    tiles = {}

    # Iterate over the fixed corners to create tiles
    for i in range(num_tiles):
        # Calculate the indices for the corners of the tile
        start_index = i * 2
        end_index = start_index + 4
        
        # Extract the corners for the current tile
        tile_corners = [fixed_corners[start_index], fixed_corners[start_index + 1], 
                        fixed_corners[start_index + 3], fixed_corners[start_index + 2]]
        
        # Assign the corners to the tile in the desired order
        tiles[f"Tile {i + 1}"] = tile_corners
    
    return tiles


def main():
    global fixed_corners

    # Read the input image
    image = cv2.imread("TestImg.png")

    # Detect and label corners
    labeled_image, _ = detect_and_label_corners(image.copy())  # Ensure we're working with a copy of the image

    # Assign tiles to corner areas
    tiles = assign_tiles_to_corners()
    print("Tiles and their corners:")
    for tile, corners in tiles.items():
        print(f"{tile}: {corners}")
        

    # Draw tiles on the image
    for tile, corners in tiles.items():
        # Convert corners to a numpy array
        corners = np.array(corners).reshape((-1, 1, 2))
        # Draw the polygon representing the tile
        cv2.polylines(labeled_image, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
        # Display the name of the tile
        text_position = (int(np.mean(corners[:, :, 0])), int(np.mean(corners[:, :, 1])))
        cv2.putText(labeled_image, tile, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image with detected and labeled corners and tiles
    cv2.imshow('Image with Labeled Corners and Tiles', labeled_image)

    # Keep the window open until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def get_fixed_corners():
    
    global fixed_corners
    return fixed_corners

def get_tiles():
    global tiles
    assign_tiles_to_corners()
    return tiles
    

if __name__ == "__main__":
    main()


