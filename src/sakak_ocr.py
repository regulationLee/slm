# original code from https://github.com/SeonminKim1/Study-OCR
# https://github.com/JaidedAI/EasyOCR

import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# import os
# import time
# import torch
# import torchvision
# from easyocr import EasyOCRDetection, EasyOCRRecognition, EasyOCRCRAFT
# import easyocr
# from PIL import Image, ImageDraw

DATA_PATH = '/home/glee/sakak/data/TSA/samples'
RESULT_PATH = '/home/glee/sakak/slm/results'

# file_name = '1-2. 소견서.jfif'
file_name = 'regulationLee00.png'
file_path = os.path.join(DATA_PATH, file_name)

result_name = '[OCR] ' + file_name
result_path = os.path.join(RESULT_PATH, result_name)

# east_model = torchvision.models.detection.east_resnet50(pretrained=True)
# resnet_model = torchvision.models.resnet50(pretrained=True)

# bilstm_model = torch.nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, 
# bidirectional=True)

# detection_module = Reader(['ko'], gpu=True, )
# EasyOCRDetection(east_model, threshold=0.5)
# recognition_module = EasyOCRRecognition(resnet_model, bilstm_model, output_dim=128)

# bboxes = detection_module.detect(image)
# text = recognition_module.recognize(image, bboxes=bboxes)

# # 시각화
# plt.imshow(image)
# for bbox in bboxes:
#     x1, y1, x2, y2 = bbox
#     plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
# plt.title(text)
# plt.savefig(result_path)
# # ```



# ocr_model = easyocr.Reader(['ko'], gpu=True)
# result = ocr_model.readtext(file_path, min_size=1)

# if len(result) > 0:
# 	image = Image.open(file_path)
# 	draw = ImageDraw.Draw(image)

# 	for detection in result:
# 		bbox, text, confidence = detection
# 		print(f"Detected text: {bbox}, {text}, (Confidence: {confidence:.2f}")

# 		draw.polygon([tuple(point) for point in bbox], outline='blue')

# 	image.save(result_path)


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms


def load_yolo_model(model_name='yolov5s'):
    # YOLOv5 모델을 로드합니다. 모델 이름은 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x' 등으로 변경 가능합니다.
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    return model

# CRNN 모델 정의 (여기서는 사용자 정의 CRNN 모델을 사용)
class CRNN(torch.nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        # CRNN 모델 구조를 정의합니다
        pass
    
    def forward(self, x):
        # 순전파 정의
        pass

crnn_model = CRNN()
# crnn_model.load_state_dict(torch.load('path/to/your/crnn_model.pth'))
crnn_model.eval()

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# CRNN을 사용한 텍스트 인식 함수
def recognize_text(image):
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        output = crnn_model(preprocessed_image)
    # 텍스트를 변환하는 부분은 모델에 따라 다를 수 있습니다
    # 여기서는 임시로 'detected text'를 사용합니다
    return 'detected text'

# 텍스트 감지 함수
def detect_text(model, image_path):
    results = model(image_path)
    boxes = results.xyxy[0].cpu().numpy()
    return boxes

# EasyOCR 클래스의 대체 클래스
class CustomEasyOCR:
    def __init__(self):
        pass
    
    def readtext(self, model, image_path):
        image = Image.open(image_path).convert('RGB')
        boxes = detect_text(model, image_path)
        texts = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_image = image.crop((x1, y1, x2, y2))
            text = recognize_text(cropped_image)
            texts.append((text, box))
        return texts

# 사용자 정의 EasyOCR 사용
reader = CustomEasyOCR()
yolo_model = load_yolo_model('yolov5s')
results = reader.readtext(yolo_model, file_path)


# 결과 시각화
image = Image.open(file_path).convert('RGB')
plt.imshow(image)
ax = plt.gca()

for text, box in results:
    x1, y1, x2, y2 = map(int, box[:4])
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(x1, y1 - 10, text, color='r', fontsize=12, weight='bold')

plt.axis('off')
plt.savefig(os.path.join(RESULT_PATH, 'result.png'))





# # YOLOv5 모델 로드
# def load_yolo_model(model_name='yolov5s'):
#     # YOLOv5 모델을 로드합니다. 모델 이름은 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x' 등으로 변경 가능합니다.
#     model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
#     return model

# # 이미지에서 텍스트 감지
# def detect_text(model, image_path):
#     # YOLOv5 모델을 사용하여 이미지에서 객체를 감지합니다.
#     results = model(file_path)
#     return results

# # 결과 시각화
# def visualize_results(image_path, results):
#     image = Image.open(image_path).convert('RGB')
#     plt.imshow(image)
#     ax = plt.gca()

#     for box in results.xyxy[0]:  # 결과는 xyxy 형식으로 제공됩니다.
#         x1, y1, x2, y2, conf, cls = map(float, box[:6])
#         rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(x1, y1 - 10, f'Class: {int(cls)} Conf: {conf:.2f}', color='r', fontsize=12, weight='bold')

#     plt.axis('off')
#     plt.savefig(os.path.join(RESULT_PATH, 'result.png'))

# # 사용 예제
# model = load_yolo_model('yolov5s')
# results = detect_text(model, file_path)
# print(results)
# visualize_results(file_path, results)
