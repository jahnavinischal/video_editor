import cv2
from enhancer import enhance_frame

VIDEO_PATH = "sample_input/blurry_noisy_input.mp4"
OUTPUT_PATH = "enhanced_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

cv2.namedWindow("Controls")

# Trackbars
def nothing(x): pass
cv2.createTrackbar("Denoise", "Controls", 10, 50, nothing)
cv2.createTrackbar("Sharpen", "Controls", 5, 30, nothing)
cv2.createTrackbar("CLAHE", "Controls", 3, 10, nothing)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    d = cv2.getTrackbarPos("Denoise", "Controls")
    s = cv2.getTrackbarPos("Sharpen", "Controls")
    c = cv2.getTrackbarPos("CLAHE", "Controls")

    enhanced = enhance_frame(frame, d, s, c)
    out.write(enhanced)

    cv2.imshow("Enhanced Video", enhanced)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video saved to: {OUTPUT_PATH}")
