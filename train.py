from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # you can change the model here, e.g., use 'yolov8x-pose'
    # Example:
    # model = YOLO('yolov8s-pose.yaml').load('yolov8s-pose.pt')
    results = model.train(data='./data/configuration.yaml', epochs=4000, imgsz=640, batch=12, patience=0, pose=25)  # Set dataset path. If there is a path error, change the path to an absolute path
