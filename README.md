# Drowsiness Detection System

A real-time drowsiness detection system using YOLOv5 to identify whether a person is awake or drowsy. This project leverages computer vision techniques and deep learning to improve road safety by alerting drivers if they show signs of drowsiness.

## Features

- **Real-Time Detection**: Uses a webcam to capture live video and detect drowsiness in real time.
- **Custom Training**: Train the model from scratch with labeled data for "awake" and "drowsy" states.
- **YOLOv5 Integration**: Utilizes the YOLOv5 model for object detection.
- **Labeling Support**: Includes integration with `labelImg` for annotating custom datasets.

## Installation

### Prerequisites

- Python 3.8+
- Pip
- GPU support (recommended for faster performance)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/PHom798/Drowsiness-Detection-System-
   cd Drowsiness-Detection-System-
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install -r yolov5/requirements.txt
   pip install pyqt5 lxml
   ```

3. Set up `labelImg` for labeling data:
   ```bash
   git clone https://github.com/tzutalin/labelImg
   cd labelImg
   pyrcc5 -o libs/resources.py resources.qrc
   ```

## Usage

### 1. Load the Pretrained Model

```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```

### 2. Real-Time Detection

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### 3. Train a Custom Model

- Collect images for training:
  ```python
  import uuid
  import os
  import time

  IMAGES_PATH = 'data/images'
  labels = ['awake', 'drowsy']
  number_imgs = 40
  for label in labels:
      for img_num in range(number_imgs):
          imgname = os.path.join(IMAGES_PATH, label + '.' + str(uuid.uuid1()) + '.jpg')
  ```

- Train the model:
  ```bash
  cd yolov5
  python train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2
  ```

### 4. Load Custom Model for Detection

```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)
```

## Project Structure

```
Drowsiness-Detection-System-
├── yolov5/                # YOLOv5 implementation
├── data/
│   ├── images/            # Collected images for training
│   └── annotations/       # Annotations for the dataset
├── labelImg/              # Tool for data labeling
└── Drowsy.ipynb           # Jupyter Notebook for implementation
```

## Future Enhancements

- Add sound alerts for real-time detection.
- Improve model accuracy with more diverse training data.
- Deploy the system as a standalone application.


## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [tzutalin/labelImg](https://github.com/tzutalin/labelImg)
