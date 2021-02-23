import cv2
import darknet
import time

# Parameters
win_title = 'YOLOv4 CUSTOM DETECTOR'
cfg_file = 'cfg/yolov4-tiny.cfg'
data_file = 'cfg/coco.data'
weight_file = 'yolov4-tiny.weights'
thre = 0.25
show_coordinates = False
show_size = (800, 800)

# Load Network
network, class_names, class_colors = darknet.load_network(
        cfg_file,
        data_file,
        weight_file,
        batch_size=1
    )

# Get Nets Input dimentions
width = darknet.network_width(network)
height = darknet.network_height(network)

# Get Webcam from cv2
cap = cv2.VideoCapture(0)

# Video Stream
while cap.isOpened():
    
    # Get current frame, quit if no frame 
    ret, frame = cap.read()

    if not ret: break

    # Calculate for FPS
    t_prev = time.time()

    # Fix image format
    frame_rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize( frame_rgb, (width, height))
    
    # convert to darknet format, save to " darknet_image "
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())    
    
    # inference
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)
    darknet.free_image(darknet_image)

    # draw bounding box
    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show Image and FPS
    fps = int(1/(time.time()-t_prev))
    cv2.rectangle(image, (5, 5), (75, 25), (0,0,0), -1)
    cv2.putText(image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(win_title, cv2.resize(image, show_size))

    # press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
