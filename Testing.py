import cv2
import numpy as np
from CornerFinder import get_fixed_corners, get_tiles
from CornerFinder import detect_and_label_corners

def main():
    print("------------------------------------")
    # Read the input image
    image = cv2.imread("C:\\Users\\saksh\\OneDrive\\Pictures\\Screenshots\\test2.png")
    
    
    # Get fixed corners from the CornerFinder module
    lab_img, fixed_corners = detect_and_label_corners(image)
    
    # print(fixed_corners)
    color = (0, 0, 255)  # Red color in BGR
    thickness = -1  # Fill the circle
    radius = 5  # Radius of the circle
    
    for i, coor in enumerate(fixed_corners):
        print(f"Corner {i+1}: {coor}")
        cv2.circle(image, (coor[0], coor[1]), radius, color, thickness)

    # Draw a red dot at the specified coordinate
    
    

    # Display the image
    cv2.imshow("Coordinate Marker", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Get tiles from the CornerFinder module
#     tiles = get_tiles()
#     print("Tiles and their corners:")
#     for tile, corners in tiles.items():
#         print(f"{tile}: {corners}")

#     # Draw tiles on the image
# # Draw tiles on the image
#     for tile, corners in tiles.items():
#         # Convert corners to a numpy array
#         corners = np.array(corners).reshape((-1, 1, 2))
#         # Draw the polygon representing the tile
#         cv2.polylines(image, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
#         # Display the name of the tile
#         text_position = (int(np.mean(corners[:, :, 0])), int(np.mean(corners[:, :, 1])))
#         cv2.putText(image, tile, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # Display the image with detected and labeled corners and tiles
#     cv2.imshow('Image with Labeled Corners and Tiles', image)


    # Keep the window open until a key is pressed
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
