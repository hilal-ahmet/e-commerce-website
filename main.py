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



@app.route('/')
def index():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)

