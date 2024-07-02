from flask import Flask, request, render_template
import cv2
import numpy as np
import io
import base64
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

app = Flask(__name__)

# YOLO modelini oluştur
yolo_model = YOLO('your/main/root/ObjectDetectionApp/yolo.pt')

# FRCNN modelini oluştur ve yükle
num_classes = 5  # Arka plan dahil sınıf sayısı
frcnn_model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
checkpoint = torch.load(r'your/main/root/ObjectDetectionApp/FastRCNN.pth')
frcnn_model.load_state_dict(checkpoint['model_state_dict'])
frcnn_model.eval()

# Görüntü dönüşümleri
transform = T.Compose([
    T.ToTensor(),
])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'images[]' not in request.files:
        return 'Görsel bulunamadı.'

    files = request.files.getlist('images[]')
    results_list = []
    results_listFrcnn = []
    for file in files:
        if file.filename == '':
            continue
        if file:
            image_bytes = file.read()
            original_image_base64, processed_image_base64, labels_and_percentages = process_image_yolo(image_bytes)
            results_list.append((original_image_base64, processed_image_base64, labels_and_percentages))
            original_image_base64, processed_image_base64, labels_and_percentages = process_image_frcnn(image_bytes)
            results_listFrcnn.append((original_image_base64, processed_image_base64, labels_and_percentages))

    return render_template('index.html', results_list=results_list, results_listFrcnn=results_listFrcnn)


@app.route('/')
def index():
    return render_template('index.html')


def process_image_yolo(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img_copy = original_img.copy()  # Kopyasını oluştur

    labels = ['kolye', 'port', 'cihaz', 'endo']
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = yolo_model(rgb_img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels_and_percentages = []
    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        label = results[0].boxes.cls[i]
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
        name = labels[label]
        if score < 0.2:
            continue
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = name + ' ' + str(format(score, '.2f'))
        cv2.putText(original_img, text, (x2, y2 - 10), font, 1.2, (255, 0, 0), 2)
        labels_and_percentages.append((name, str(format(score, '.2f'))))

    # Orijinal görüntüyü base64 formatına dönüştür
    _, original_img_encoded = cv2.imencode('.jpg', original_img_copy)
    original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

    # İşlenmiş görüntüyü base64 formatına dönüştür
    _, processed_img_encoded = cv2.imencode('.jpg', original_img)
    processed_img_base64 = base64.b64encode(processed_img_encoded).decode('utf-8')

    return original_img_base64, processed_img_base64, labels_and_percentages


def process_image_frcnn(image_bytes):
    # Görüntüyü yükle ve dönüştür
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image)

    # Görüntüyü modele gönder ve tahminleri al
    with torch.no_grad():
        prediction = frcnn_model([image_tensor])[0]

    labels_and_percentages = []
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'kolye', 'port', 'cihaz', 'endo']
    COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

    original_img = np.array(image)
    for i, (box, label, score) in enumerate(zip(prediction['boxes'], prediction['labels'], prediction['scores'])):
        if score > 0.5:  # Belirli bir eşik değer üzerinde olan tahminleri görselleştir
            box = box.int().numpy()
            label = label.item()
            score = score.item()
            color = COLORS[label]
            cv2.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
            text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            cv2.putText(original_img, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
            labels_and_percentages.append((COCO_INSTANCE_CATEGORY_NAMES[label], f"{score:.2f}"))

    # Orijinal görüntüyü base64 formatına dönüştür
    original_img_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # İşlenmiş görüntüyü base64 formatına dönüştür
    _, processed_img_encoded = cv2.imencode('.jpg', original_img)
    processed_img_base64 = base64.b64encode(processed_img_encoded).decode('utf-8')

    return original_img_base64, processed_img_base64, labels_and_percentages


if __name__ == '__main__':
    app.run(debug=True)

