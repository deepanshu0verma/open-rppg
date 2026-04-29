import cv2
import time

cap = cv2.VideoCapture(0)

# 1. Automatically get the camera's actual width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

# 2. Use 'avc1' (H.264) which is much more stable on Mac than 'mp4v'
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))

print(f"Camera Resolution: {width}x{height}")
print("Recording starts in 3 seconds... Sit still and center your face!")
time.sleep(3)
print("RECORDING... Stay still for 60 seconds. Press 'q' to stop early.")

start_time = time.time()
while(int(time.time() - start_time) < 60):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording...', frame)
        # Standard Mac exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frame")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! 'test_video.mp4' created with resolution {width}x{height}")