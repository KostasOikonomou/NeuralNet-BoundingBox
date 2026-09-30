from ultralytics import YOLO
import torch

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda":
        print("gpu")
        
    # load pre-trained model (YOLOv8 nano)
    model = YOLO("yolov8n.pt")

    # Training on my data
    results = model.train(
        data="my_dataset_2/data.yaml",
        epochs=15,
        imgsz=1024, #pixel size
        batch=8,
        device=device,
        exist_ok = True,
        hsv_h=0.015,  # αποχρωση
        hsv_s=0.7,    # κορεσμος
        hsv_v=0.4     # φωτεινοτητα
    )

#if file is executed by the user, run function main()
if __name__=="__main__":
    main()

# Training is saved in runs/detect/train/weights
# best.pt --> best version of the model with highest accuracy
# last.pt --> last epoch model
# future use with: model = YOLO(runs/detect/train/weights/best.pt)
# and model.predict(...)