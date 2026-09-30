import cv2
from ultralytics import YOLO

def test_on_image():

    model = YOLO("runs/detect/train/weights/best.pt")

    # use pictures from "my_dataset/valid/images"
    # conf=0.25 --> confidence threshold to show bounding box
    # exist_ok = True --> don't create new file predict each time, use the same "runs/detect/predict/" (overwriting)
    results = model.predict(source="my_dataset_2/valid/images",
                            save=True,
                            conf=0.25,
                            exist_ok=True)

    print("results have been saved in runs/detect/predict/")

if __name__ == "__main__":
    test_on_image()