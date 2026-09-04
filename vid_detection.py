import cv2
import numpy as np


# -----------------------------------
# Region of Interest
# -----------------------------------
def roi(image, vertices):
    mask = np.zeros_like(image)

    cv2.fillPoly(mask, vertices, 255)

    cropped_img = cv2.bitwise_and(image, mask)

    return cropped_img


# -----------------------------------
# Draw Hough Lines
# -----------------------------------
def draw_lines(image, hough_lines):

    if hough_lines is None:
        return image

    for line in hough_lines:

        x1, y1, x2, y2 = line

        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return image


# -----------------------------------
# Process Frame
# -----------------------------------
def process(img):

    if img is None:
        return None

    height = img.shape[0]
    width = img.shape[1]

    # Define ROI
    roi_vertices = [
        (0, 650),
        (int(2 * width / 3), int(2 * height / 3)),
        (width, 1000)
    ]

    # Grayscale
    gray_img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Dilation
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    gray_img = cv2.dilate(
        gray_img,
        kernel
    )

    # Canny Edge Detection
    canny = cv2.Canny(
        gray_img,
        130,
        220
    )

    # ROI
    roi_img = roi(
        canny,
        np.array(
            [roi_vertices],
            np.int32
        )
    )

    # Hough Lines
    lines = cv2.HoughLinesP(
        roi_img,
        1,
        np.pi / 180,
        threshold=10,
        minLineLength=15,
        maxLineGap=2
    )

    # Draw lines
    final_img = draw_lines(
        img,
        lines
    )

    return final_img


# -----------------------------------
# Open Input Video
# -----------------------------------
video_path = "./Data/lane_vid2.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("ERROR: Could not open:")
    print(video_path)

    exit()


# -----------------------------------
# Video Properties
# -----------------------------------
frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0


# -----------------------------------
# Output Video
# -----------------------------------
fourcc = cv2.VideoWriter_fourcc(
    *"XVID"
)

output_path = "lane_detection.avi"

saved_frame = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not saved_frame.isOpened():

    print("ERROR: Could not create output video.")

    cap.release()

    exit()


# -----------------------------------
# Process Video
# -----------------------------------
while True:

    ret, frame = cap.read()

    # Video finished
    if not ret:
        print("Video processing completed.")
        break

    # Process frame
    frame = process(frame)

    if frame is None:
        print("ERROR: Frame processing failed.")
        break

    # Save frame
    saved_frame.write(frame)

    # Display
    cv2.imshow(
        "Road Lane Detection",
        frame
    )

    # ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:
        print("Processing stopped by user.")
        break


# -----------------------------------
# Cleanup
# -----------------------------------
cap.release()

saved_frame.release()

cv2.destroyAllWindows()

print("Output saved as:", output_path)
