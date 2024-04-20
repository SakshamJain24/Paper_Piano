# import cv2
# import numpy as np

# def main():
#     # Initialize the camera
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Read the frame from the camera
#         _, frame = cap.read()

#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply Canny edge detector
#         edges = cv2.Canny(blurred, threshold1=50, threshold2=200)

#         # Call the edgeCounter function and pass the frame and edges
#         edgeCounter(frame, edges)

#         # Display the frame with labeled edges
#         cv2.imshow('Labeled Edges', frame)

#         # Break the loop with 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# def edgeCounter(frame, edges):
#     # Use Hough Transform to detect lines
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=250)


#     # Check if 'lines' is not None and more than one line is detected
#     if lines is not None and len(lines) > 1:
#         num_lines = len(lines)

#         # Sort lines by their starting x-coordinate
#         sorted_lines = sorted(lines, key=lambda x: x[0][0])

#         # Normalize the line indices
#         for i, line in enumerate(sorted_lines):
#             normalized_value = i / (num_lines - 1)
#             # Draw the line on the frame
#             cv2.line(frame, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
#             # Label the line
#             cv2.putText(frame, f"{normalized_value:.2f}", (line[0][0], line[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     elif lines is not None:
#         # If only one line is detected, label it as 0.5
#         line = lines[0]
#         cv2.line(frame, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
#         cv2.putText(frame, "0.5", (line[0][0], line[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     else:
#         print("No lines were detected.")

# main()

# import cv2
# import numpy as np

# def main():
#     # Initialize the camera
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Read the frame from the camera
#         _, frame = cap.read()

#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply Canny edge detector
#         edges = cv2.Canny(blurred, threshold1=50, threshold2=200)

#         # Find contours
#         contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Sort contours from left to right
#         contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

#         # Initialize tile counter
#         tile_counter = 1

#         # Loop over the contours
#         for c in contours:
#             # Approximate the contour
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.04 * peri, True)

#             # If the approximated contour has four points, we can assume we have found a rectangle
#             if len(approx) == 4:
#                 # Draw the contour and label the tile
#                 cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
#                 x, y, w, h = cv2.boundingRect(approx)
#                 cv2.putText(frame, f"Tile {tile_counter}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 tile_counter += 1

#         # Display the frame with labeled tiles
#         cv2.imshow('Labeled Tiles', frame)

#         # Break the loop with 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# main()
import cv2
import numpy as np

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the camera
        _, frame = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detector
        # Try adjusting the threshold values for better edge detection
        edges = cv2.Canny(blurred, threshold1=30, threshold2=150)  # Adjust these values

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        # Initialize tile counter
        tile_counter = 0

        # Loop over the contours
        for c in contours:
            # Approximate the contour
            # Adjust the approximation accuracy as needed
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Adjust the approximation factor

            # If the approximated contour has four points, we can assume we have found a rectangle
            if len(approx) == 4:
                # Draw the contour and label the tile
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.putText(frame, f"Tile {tile_counter}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                tile_counter += 1

        # Display the frame with labeled tiles
        cv2.imshow('Labeled Tiles', frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
