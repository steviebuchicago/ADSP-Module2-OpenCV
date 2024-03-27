import cv2

# Callback function for trackbar (required but not used)
def nothing(x):
    pass

# Initialize webcam
cap = cv2.VideoCapture(1)

# Create a window for the HSV adjustments
cv2.namedWindow('HSV Adjustments')
cv2.resizeWindow('HSV Adjustments', 300, 300)

# Create trackbars for HSV range adjustments
# Grouping H, S, V Min and Max's together
cv2.createTrackbar('H Min', 'HSV Adjustments', 0, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Adjustments', 179, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Adjustments', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Adjustments', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Adjustments', 0, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Adjustments', 255, 255, nothing)
cv2.createTrackbar('Min Contour Area', 'HSV Adjustments', 1000, 10000, nothing)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read the HSV range values from the trackbars
        h_min = cv2.getTrackbarPos('H Min', 'HSV Adjustments')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Adjustments')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Adjustments')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Adjustments')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Adjustments')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Adjustments')

        # Create the HSV range mask
        lower_hsv = (h_min, s_min, v_min)
        upper_hsv = (h_max, s_max, v_max)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Apply the mask to extract the color
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find and draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'HSV Adjustments')
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

        # Display the original frame with contours and the masked frame
        cv2.imshow('Frame with Contours', frame)
        cv2.imshow('HSV Filtered', result)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
