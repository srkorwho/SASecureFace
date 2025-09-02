import sys
import cv2
import numpy as np
import mss
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.showFullScreen()
        self.boxes = []

    def update_boxes(self, boxes):
        self.boxes = boxes
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        for (x, y, w, h, is_match) in self.boxes:
            if is_match:
                pen = QtGui.QPen(QtGui.QColor(255, 0, 255), 3)
            else:
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 3)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)


class FaceDetector(QtCore.QThread):
    boxes_detected = QtCore.pyqtSignal(list)

    def __init__(self, text_features):
        super().__init__()
        self.running = True
        self.model = YOLO("yolov8n-face.pt")
        self.text_features = text_features

    def run(self):
        sct = mss.mss()
        monitor = sct.monitors[1]

        while self.running:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            results = self.model(frame, verbose=False)
            boxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    extra_top = int(h * 0.6)
                    extra_side = int(w * 0.3)
                    x1_ext = max(0, x1 - extra_side)
                    x2_ext = min(frame.shape[1], x2 + extra_side)
                    y1_ext = max(0, y1 - extra_top)
                    y2_ext = min(frame.shape[0], y2)
                    face_crop = frame[y1_ext:y2_ext, x1_ext:x2_ext]
                    if face_crop.size == 0:
                        continue

                    pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_tensor)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        similarity = torch.cosine_similarity(image_features, self.text_features).item()

                    is_match = similarity > 0.25
                    boxes.append((x1_ext, y1_ext, x2_ext - x1_ext, y2_ext - y1_ext, is_match))

            self.boxes_detected.emit(boxes)

    def stop(self):
        self.running = False
        self.wait()


class App(QtWidgets.QApplication):
    def __init__(self, sys_argv, text_features):
        super().__init__(sys_argv)
        self.overlay = Overlay()
        self.detector = FaceDetector(text_features)
        self.detector.boxes_detected.connect(self.overlay.update_boxes)
        self.detector.start()

    def quit(self):
        self.detector.stop()
        super().quit()


if __name__ == "__main__":
    prompt = input("Enter Prompt: ")
    text_tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    app = App(sys.argv, text_features)
    sys.exit(app.exec_())
