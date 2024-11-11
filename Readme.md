# 🦠 Real Time Mask detection with YOLOv5

⚠️ **Alert** : This project was completed in 2022.

This project implements a mask detection system using the YOLOv5 object detection model. 

![Demo](assets/demo.gif)

## Dataset
The model is fine-tuned on a custom dataset containing images of people with and without face masks, enabling it to detect individuals with and without masks in real-time or from images and videos.

The dataset is available here : [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

You can download the zip file and unzip it in the data/ folder:

```bash
unzip archive.zip -d data/
```

## How to setup the project ?

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## Fine-tune the YOLOv5 model

### 1. Setup the data

To train the model, you first have to prepare the data. YOLOv5 requires a data format that is not that of the downloaded dataset. To convert the data to the correct format, you can run the notebook data_preparation.ipynb. You should have something like this : 

```kotlin
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── lables/
│       ├── image1.txt
│       ├── image2.txt
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── lables/
│       ├── image1.txt
│       ├── image2.txt
```

### 2. Train the model

Run this command :

```bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data mask_data.yml --weights yolov5s.pt --name mask_yolov5
```

- **--img 640:** Sets the input image size to 640x640 pixels. Larger image sizes can increase accuracy but require more computational power.
- **--batch 16:** Specifies a batch size of 16 images per training iteration. Adjust based on GPU memory capacity.
- **--epochs 50:** Sets the training to run for 50 epochs. More epochs can improve accuracy but will take longer.
- **--data mask_data.yaml:** Points to the dataset configuration file, mask_data.yaml, which specifies the paths for training and validation images, the number of classes, and class names.
- **--weights yolov5s.pt:** Loads the YOLOv5 small model (yolov5s.pt) pretrained on the COCO dataset. Using pretrained weights speeds up training by leveraging knowledge from the COCO dataset.
- **--name mask_yolov5:** Sets the name for the training run. Results, including logs and weights, are saved in runs/train/mask_yolov5.

### 3. Evaluate the model

Run this command :

```bash
python yolov5/val.py --weights yolov5/runs/train/mask_yolov5/weights/best.pt --data mask_data.yaml --img 640
```

- **--weights yolov5/runs/train/mask_yolov5/weights/best.pt:** Loads the best-performing weights from the training process. These weights are automatically saved at this path during training.
- **--data mask_data.yaml:** Points to the same dataset configuration file used in training.
- **--img 640:** Sets the image size for validation to 640x640 pixels (matching the training size).

### 3. Export the model

Run this command :

```bash
python yolov5/export.py --weights yolov5/runs/train/mask_yolov5/weights/best.pt --include onnx torchscript
```

- **--weights yolov5/runs/train/mask_yolov5/weights/best.pt:** Loads the best weights from the training process to export.
- **--include onnx torchscript:** Specifies the export formats.

### 4. Real time detection on webcam

Run this command :

```bash
python yolov5/detect.py --weights yolov5s.pt --source 0 --img 640 --conf 0.25
```