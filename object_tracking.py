
import cv2
import torch
from ultralytics import YOLO
import random
import numpy as np
from collections import defaultdict, deque

# Live Stream with OpenCV
# 1) cap = cv.VideoCapture(video_source)
# 2) cap.read() = (success,frame)
# 3) results = model.predict(...,source=frame,...)
# 4) annotated_frame = result[0].plot()
# 5) cv2.show("name",annotated_frame)

def process_video():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working with: ",device)

    # loading trained model from another execution
    model_path = "runs/detect/train/weights/best.pt"
    model = YOLO(model_path)

    num_videos = 4
    list_of_videos=["input_video"+str(i+1)+".mp4" for i in range(num_videos)]
    video_source = random.choice(list_of_videos)

    # cap --> video-reading object
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: no opened source: ", video_source)
        return

    print("Processing video... (press 'q' to exit)")

    # editing size of the window 
    # creates/edits the window's attributes before the image is shown
    cv2.namedWindow("YOLOv8 Video Analysis", cv2.WINDOW_NORMAL) # WINDOW-NORMAL --> flag for resizable window
    # width=1024, height=576 (16:9 analysis)
    cv2.resizeWindow("YOLOv8 Video Analysis", 1024, 576)

    # **********************************************************
    # when the 31st object is appended, the 1st gets deleted
    # deafultdict(): when there is no such key, it creates new object with default value and that key
    # default value--> queue of the 30 objects
    track_history = defaultdict(lambda: deque(maxlen=100))
    # **********************************************************


    # main processing loop
    while cap.isOpened():
        # reading next frame
        # success (read or not) --> T/F
        # frame --> image as NumPy Array
        success, frame = cap.read()
        if not success:
            print("Video ended or reading error")
            break

        # **********************************************************

        results = model.track(
            source=frame,
            imgsz=640,
            conf=0.15,
            device=device,
            tracker="botsort.yaml",  # tracking algorithm
            persist=True,  # keeps memory, consecutive frames
            verbose = False  # no spam terminal messages
        )
        # **********************************************************

        # **********************************************************

        # .plot() : function of YOLO, draws the bounding boxes
        annotated_frame = results[0].plot()

        # Detect objects with IDs
        # results[0].boxes--> object for this frame
        # results[0].boxes.id--> track IDs given for object
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # (x1,y1,x2,y2) and Track IDs
            boxes = results[0].boxes.xywh.cpu().numpy()  # bounding box in the form of [x_center,y_center,width,height], converted to numpy
            track_ids = results[0].boxes.id.int().cpu().tolist() # unique IDs-->list of integers
            cls_ids = results[0].boxes.cls.int().cpu().tolist() #classes --> list of integers (e.g. 0=player, 32=ball)

            # Trajectory plotting for each ID
            # zip-->connects each box with the proper ID
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (int(x), int(y))
                
                # Save the center in the ID's tracking history
                # track_history-->queue consisting of tuples (x,y)
                track_history[track_id].append(center)

                # Drawing Trajectory Line (with >2 points)
                if len(track_history[track_id])>1:
                    # np.hstack= horizontal stack of (x,y)-->[x1,y1,x2,y2,x3,y3], and reshape to [N,1,2] -->[[[10,20]],[[12,22]],[[15,25]]]
                    points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                    # draws continuous line ([points]-->the spots of the line)
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=8)

        # **********************************************************




        

        # opens page with name "YOLOv8 Video Analysis", shows the processed frame
        cv2.imshow("YOLOv8 Video Analysis", annotated_frame)


        # cv2.waitKey(1) --> wait 1 ms in order to update the widnow
        # if user presses 'q' stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()  # releases the video from the memory
    cv2.destroyAllWindows()  


if __name__=="__main__":
    process_video()