import cv2
import pyautogui

# Define the lower and upper HSV color ranges for the yellow pen
hue_min, saturation_min, value_min = 25, 100, 100
hue_max, saturation_max, value_max = 35, 255, 255

# Constants for finger IDs
INDEX_FINGER_ID = 8

# Smoothing factor for cursor movement
SMOOTHING = 0.5

# Function to track the yellow pen
def track_pen(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV color ranges for the pen
    lower_color = (hue_min, saturation_min, value_min)
    upper_color = (hue_max, saturation_max, value_max)

    # Threshold the HSV image to get only the pen color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Reduce the kernel size
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

# Function to detect hand gestures and move the cursor
def detect_gestures(hand, frame, cursor_position):
    # Get the position of the index finger
    index_finger = hand[INDEX_FINGER_ID]

    # Get the coordinates of the finger
    height, width, _ = frame.shape
    index_x, index_y = int(index_finger[0] * width), int(index_finger[1] * height)

    # Move the cursor based on the index finger position
    cursor_x, cursor_y = cursor_position
    cursor_x = int(SMOOTHING * cursor_x + (1 - SMOOTHING) * index_x)
    cursor_y = int(SMOOTHING * cursor_y + (1 - SMOOTHING) * index_y)

    pyautogui.moveTo(cursor_x, cursor_y)

    # Check if the index finger is lifted (y-coordinate value)
    index_finger_lifted = index_y < height - 20  # Adjust the threshold based on your hand position

    # Perform mouse actions based on the detected gestures
    if index_finger_lifted:
        pyautogui.mouseUp()  # Release the mouse button if the index finger is lifted
    else:
        pyautogui.mouseDown()  # Press the mouse button if the index finger is down

    return cursor_x, cursor_y

# Main program loop
cap = cv2.VideoCapture(0)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load the Haar cascade classifier for hand detection using absolute path
hand_cascade_file = "C:/path/to/hand.xml"  # Replace with the absolute path to hand.xml on your system
hand_cascade = cv2.CascadeClassifier(hand_cascade_file)

# Initial cursor position
cursor_position = (width // 2, height // 2)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally (optional, depending on your camera setup)
    frame = cv2.flip(frame, 1)

    # Track the yellow pen in the frame
    frame = track_pen(frame)

    # Convert the frame to grayscale for hand detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands using the cascade classifier
    # Load the Haar Cascade classifier for hand detection
    hand_cascade = cv2.CascadeClassifier('hand.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for better performance
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect hands in the frame
        hands = hand_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected hands
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If a hand is detected, process hand landmarks to detect gestures and move the cursor
    if len(hands) > 0:
        hand = hands[0]  # We assume only one hand is detected
        cursor_position = detect_gestures(hand, frame, cursor_position)

    # Display the frame
    cv2.imshow("Pen Tracking", frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
