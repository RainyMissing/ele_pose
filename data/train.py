from ultralytics import YOLO

# Load a model
model = YOLO(model='yolov8x.yaml', task='detect', data_dir='./theos_data')

# Train the model
model.train(data='./coco8.yaml', epochs=2)