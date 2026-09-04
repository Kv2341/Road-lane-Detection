import cv2
import numpy as np

# Define Region of Interest (ROI)
def roi(image, vertices):
    mask = np.zeros_like(image)

    mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)

    masked_img = cv2.bitwise_and(image, mask)

    return masked_img

# Draw Hough Lines on Image

def draw_lines(lines, image):

    # If no lines are detected
    if lines is None:
        return image

    for line in lines:

        # HoughLinesP returns [x1, y1, x2, y2]
        x1, y1, x2, y2 = line

        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return image

# Process One Video Frame

def process(img):

    # Check if frame is valid
    if img is None:
        return None

    try:

        # Get frame dimensions
        h, w, _ = img.shape

        # Define ROI vertices

        roi_vertices = [
            (200, h),
            (w // 2, 2 * h // 3),
            (w - 100, h)
        ]

        # Convert to Grayscale

        gray_img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Apply Dilation

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        gray_img = cv2.dilate(
            gray_img,
            kernel=kernel
        )

        # Canny Edge Detection

        canny = cv2.Canny(
            gray_img,
            60,
            255
        )

        # Apply Region of Interest

        roi_image = roi(
            canny,
            np.array(
                [roi_vertices],
                np.int32
            )
        )

        # Hough Line Detection

        hough_lines = cv2.HoughLinesP(
            roi_image,
            1,
            np.pi / 180,
            40,
            minLineLength=10,
            maxLineGap=5
        )

        # Draw Detected Lines

        final_img = draw_lines(
            hough_lines,
            img
        )

        return final_img

    except Exception as e:

        print("Error while processing frame:", e)

        return img

# Open Input Video

video_path = "./Data/Manhattan_Trim.mp4"

cap = cv2.VideoCapture(video_path)

# Check Video

if not cap.isOpened():

    print("Error: Could not open input video.")

    exit()

# Get Video Properties

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

# Use 30 FPS if FPS is unavailable
if fps <= 0:
    fps = 30.0

# Create Output Video

fourcc = cv2.VideoWriter_fourcc(
    *"XVID"
)

output_path = "Manhattan_detection.avi"

saved_frame = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

# Check Output Video

if not saved_frame.isOpened():

    print("Error: Could not create output video.")

    cap.release()

    exit()

# Process Video Frame by Frame

while cap.isOpened():

    # Read one frame
    ret, frame = cap.read()

    # Stop when video ends
    if not ret:

        print("Video processing completed.")

        break

    # Process current frame
    processed_frame = process(frame)

    # Check processed frame
    if processed_frame is None:

        print("Error: Could not process frame.")

        break

    # Save processed frame
    saved_frame.write(processed_frame)

    # Display processed frame
    cv2.imshow(
        "Road Lane Detection",
        processed_frame
    )

    # Press ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:

        print("Processing stopped by user.")

        break

# Release Resources

cap.release()

saved_frame.release()

cv2.destroyAllWindows()

print("Output saved as:", output_path)
