import cv2

# Callback function for trackbar (required but not used)
def nothing(x):
    pass

# Initialize webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Cannot access webcam")

# Create a window for the HSV and Shape adjustments
cv2.namedWindow('Adjustments')
cv2.resizeWindow('Adjustments', 500, 375)

# Create trackbars for HSV range adjustments
cv2.createTrackbar('H Min', 'Adjustments', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Adjustments', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Adjustments', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Adjustments', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Adjustments', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Adjustments', 255, 255, nothing)
cv2.createTrackbar('Min Contour Area', 'Adjustments', 1000, 10000, nothing)
cv2.createTrackbar('Polygonal Sides', 'Adjustments', 3, 10, nothing)  # For len(approx)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read the HSV range and shape values from the trackbars
        h_min = cv2.getTrackbarPos('H Min', 'Adjustments')
        h_max = cv2.getTrackbarPos('H Max', 'Adjustments')
        s_min = cv2.getTrackbarPos('S Min', 'Adjustments')
        s_max = cv2.getTrackbarPos('S Max', 'Adjustments')
        v_min = cv2.getTrackbarPos('V Min', 'Adjustments')
        v_max = cv2.getTrackbarPos('V Max', 'Adjustments')
        min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Adjustments')
        sides = cv2.getTrackbarPos('Polygonal Sides', 'Adjustments')

        # Create the HSV range mask
        lower_hsv = (h_min, s_min, v_min)
        upper_hsv = (h_max, s_max, v_max)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Apply the mask to the frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours and draw those that match the criteria
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Approximate contour shape and analyze its vertices
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == sides and cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

        # Display the original frame with contours and the filtered result
        cv2.imshow('Frame with Contours', frame)
        cv2.imshow('HSV Filtered', result)

        # Break the loop with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
