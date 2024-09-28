import cv2
import numpy as np

# Define the lower and upper HSV color ranges for a blue pen
hue_min, saturation_min, value_min = 90, 100, 100
hue_max, saturation_max, value_max = 120, 255, 255

# Function to track the pen
def track_pen(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV color ranges for the pen
    lower_color = np.array([hue_min, saturation_min, value_min])
    upper_color = np.array([hue_max, saturation_max, value_max])

    # Threshold the HSV image to get only the pen color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the pen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if len(contours) > 0:
        # Find the largest contour (the pen)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the centroid of the largest contour
        M = cv2.moments(largest_contour)
        centroid_x = int(M['m10'] / (M['m00'] + 1e-5))  # Adding a small value to avoid division by zero
        centroid_y = int(M['m01'] / (M['m00'] + 1e-5))

        # Draw a circle at the centroid of the pen
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

    return frame

# Main program loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally (optional, depending on your camera setup)
    frame = cv2.flip(frame, 1)

    # Track the pen in the frame
    frame = track_pen(frame)

    # Display the frame
    cv2.imshow("Pen Tracking", frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
