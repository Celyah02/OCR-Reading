# printed_text_scanner.py
import sys
import os
import time
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
import pytesseract
from PyQt5 import QtCore, QtGui, QtWidgets

# --- UPDATE THIS PATH if Tesseract is not in PATH on Windows ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

SAVE_DIR = "scanned_texts"
os.makedirs(SAVE_DIR, exist_ok=True)


def cv2_to_qpixmap(cv_img):
    """Convert an OpenCV image (BGR or grayscale) to QPixmap."""
    if cv_img is None:
        return QtGui.QPixmap()
    if len(cv_img.shape) == 2:
        height, width = cv_img.shape
        bytes_per_line = width
        q_img = QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    else:
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QtGui.QImage(cv_img_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(q_img)


class ImageLabel(QtWidgets.QLabel):
    """QLabel that supports click-and-drag ROI selection and shows a rectangle overlay."""

    roi_changed = QtCore.pyqtSignal(tuple)  # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.pixmap_orig = None
        self.dragging = False
        self.start_pos = None
        self.end_pos = None
        self._scale = 1.0
        self._offset = QtCore.QPoint(0, 0)

        # store last displayed pixmap so we can re-draw overlay
        self.display_pixmap = None

    def setPixmap(self, pm: QtGui.QPixmap):
        self.pixmap_orig = pm
        super().setPixmap(pm)
        self.display_pixmap = pm

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self.pixmap() is None:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self.dragging = True
            self.start_pos = ev.pos()
            self.end_pos = ev.pos()
            self.update()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self.dragging:
            self.end_pos = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self.dragging:
            self.dragging = False
            self.end_pos = ev.pos()
            self.update()
            roi = self.get_roi_coords()
            if roi:
                self.roi_changed.emit(roi)

    def paintEvent(self, ev: QtGui.QPaintEvent):
        super().paintEvent(ev)
        if self.pixmap() is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # draw selection rectangle if dragging or selected
        if self.start_pos and self.end_pos:
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            rect = QtCore.QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect.normalized())

    def get_roi_coords(self):
        """Return ROI in source image coordinates: (x, y, w, h) or None."""
        if not self.pixmap_orig or not self.start_pos or not self.end_pos:
            return None

        pm = self.pixmap()
        # compute scaling from pixmap size to label widget size
        label_w, label_h = self.width(), self.height()
        pixmap_w, pixmap_h = pm.width(), pm.height()

        if pixmap_w == 0 or pixmap_h == 0:
            return None

        # Determine top-left location of pixmap inside label (centered)
        x_offset = max((label_w - pixmap_w) // 2, 0)
        y_offset = max((label_h - pixmap_h) // 2, 0)

        # compute rectangle in pixmap coordinates
        rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
        rect.translate(-x_offset, -y_offset)

        # clip to pixmap
        rect = rect.intersected(QtCore.QRect(0, 0, pixmap_w, pixmap_h))
        if rect.width() <= 0 or rect.height() <= 0:
            return None

        # scale to original image size (pixmap created from original size, so one-to-one)
        # If pixmap was scaled to fit QLabel, map rect proportionally:
        orig_pm_w, orig_pm_h = self.pixmap_orig.width(), self.pixmap_orig.height()
        scale_x = orig_pm_w / pixmap_w
        scale_y = orig_pm_h / pixmap_h

        x = int(rect.x() * scale_x)
        y = int(rect.y() * scale_y)
        w = int(rect.width() * scale_x)
        h = int(rect.height() * scale_y)
        return (x, y, w, h)


class OCRWorker(QtCore.QObject):
    """Runs OCR in a background thread to avoid freezing the UI."""
    finished = QtCore.pyqtSignal(str, np.ndarray)  # text, overlay_image

    def __init__(self, image: np.ndarray, do_overlay=True):
        super().__init__()
        self.image = image.copy()
        self.do_overlay = do_overlay

    @QtCore.pyqtSlot()
    def run(self):
        # preprocessing: grayscale + Otsu
        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = self.image.copy()
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # use pytesseract to get text and bounding boxes
        custom_config = "--psm 6"  # assume a block of text
        text = pytesseract.image_to_string(thresh, config=custom_config)

        overlay = None
        if self.do_overlay:
            overlay = self.image.copy()
            # get word-level data
            data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['level'])
            for i in range(n_boxes):
                txt = data['text'][i].strip()
                if txt != "":
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # put small text label
                    cv2.putText(overlay, txt, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # emit result
        self.finished.emit(text, overlay if overlay is not None else self.image)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Printed Text Scanner (PyQt5 + PyTesseract)")
        self.setGeometry(120, 60, 1200, 700)

        # central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: image area
        left_v = QtWidgets.QVBoxLayout()
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #333;")
        left_v.addWidget(self.image_label)

        # Camera controls
        cam_h = QtWidgets.QHBoxLayout()
        self.btn_start_cam = QtWidgets.QPushButton("Start Camera")
        self.btn_start_cam.clicked.connect(self.toggle_camera)
        cam_h.addWidget(self.btn_start_cam)

        self.btn_capture = QtWidgets.QPushButton("Capture Frame")
        self.btn_capture.clicked.connect(self.capture_frame)
        cam_h.addWidget(self.btn_capture)

        self.btn_load = QtWidgets.QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        cam_h.addWidget(self.btn_load)

        left_v.addLayout(cam_h)

        # Right: controls + text result
        right_v = QtWidgets.QVBoxLayout()

        # OCR control buttons
        btn_h = QtWidgets.QHBoxLayout()
        self.btn_run_ocr = QtWidgets.QPushButton("Run OCR (ROI or Full)")
        self.btn_run_ocr.clicked.connect(self.run_ocr)
        btn_h.addWidget(self.btn_run_ocr)

        self.btn_overlay = QtWidgets.QPushButton("Show Overlay")
        self.btn_overlay.clicked.connect(self.show_last_overlay)
        btn_h.addWidget(self.btn_overlay)

        self.btn_save_text = QtWidgets.QPushButton("Save Text")
        self.btn_save_text.clicked.connect(self.save_text)
        btn_h.addWidget(self.btn_save_text)

        right_v.addLayout(btn_h)

        # Text results
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(False)
        self.text_edit.setPlaceholderText("OCR extracted text will appear here...")
        right_v.addWidget(self.text_edit)

        # Status / instructions
        self.status_label = QtWidgets.QLabel("Instructions: Load image or start camera. Drag on image to select ROI. Click 'Run OCR'.")
        right_v.addWidget(self.status_label)

        # Add to main layout
        layout.addLayout(left_v, 3)
        layout.addLayout(right_v, 2)

        # internal state
        self.cv_image = None  # current image (BGR)
        self.last_overlay = None
        self.camera = None
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self._grab_frame)
        self.is_camera_running = False
        self.current_roi = None

        # connect ROI signal
        self.image_label.roi_changed.connect(self.on_roi_changed)

        # thread handling for OCR
        self.ocr_thread = None
        self.ocr_worker = None

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        self.set_image(img)
        self.status_label.setText(f"Loaded image: {os.path.basename(path)}")

    def set_image(self, cv_img):
        """Set image (BGR numpy)."""
        self.cv_image = cv_img.copy()
        pm = cv2_to_qpixmap(self.cv_image)
        # scale pixmap to fit label while keeping aspect ratio
        pm_scaled = pm.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pm_scaled)
        self.last_overlay = None
        self.current_roi = None

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # rescale pixmap to label size
        if self.cv_image is not None:
            pm = cv2_to_qpixmap(self.cv_image)
            pm_scaled = pm.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pm_scaled)
        elif self.image_label.pixmap() is not None:
            self.image_label.setPixmap(self.image_label.pixmap().scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def toggle_camera(self):
        if not self.is_camera_running:
            # try to open default camera
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QtWidgets.QMessageBox.warning(self, "Camera Error", "Cannot open camera device.")
                return
            self.is_camera_running = True
            self.btn_start_cam.setText("Stop Camera")
            self.camera_timer.start(30)
            self.status_label.setText("Camera started. Press 'Capture Frame' to freeze a frame.")
        else:
            self.camera_timer.stop()
            if self.camera:
                self.camera.release()
                self.camera = None
            self.is_camera_running = False
            self.btn_start_cam.setText("Start Camera")
            self.status_label.setText("Camera stopped.")

    def _grab_frame(self):
        if not self.camera:
            return
        ret, frame = self.camera.read()
        if not ret:
            return
        # overlay simple instructions on live feed
        disp = frame.copy()
        cv2.putText(disp, "Live Camera - press 'Capture Frame' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # convert and show
        pm = cv2_to_qpixmap(disp)
        pm_scaled = pm.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pm_scaled)

    def capture_frame(self):
        if not self.is_camera_running or self.camera is None:
            QtWidgets.QMessageBox.information(self, "Info", "Start the camera first.")
            return
        ret, frame = self.camera.read()
        if not ret:
            QtWidgets.QMessageBox.warning(self, "Warning", "Failed to capture frame.")
            return
        # stop camera and show captured image
        self.toggle_camera()
        self.set_image(frame)
        self.status_label.setText("Captured a frame. Select ROI or run OCR on full image.")

    def on_roi_changed(self, roi_tuple):
        """roi_tuple: (x, y, w, h) in pixmap/original coordinates"""
        self.current_roi = roi_tuple
        self.status_label.setText(f"ROI set: x={roi_tuple[0]} y={roi_tuple[1]} w={roi_tuple[2]} h={roi_tuple[3]}")

    def run_ocr(self):
        if self.cv_image is None:
            QtWidgets.QMessageBox.information(self, "Info", "Load an image or capture a frame first.")
            return

        # select ROI or full image
        if self.current_roi:
            x, y, w, h = self.current_roi
            roi_img = self.cv_image[y:y + h, x:x + w].copy()
        else:
            roi_img = self.cv_image.copy()

        # start OCR thread
        self.btn_run_ocr.setEnabled(False)
        self.status_label.setText("Running OCR... (this may take a moment)")
        self._start_ocr_thread(roi_img)

    def _start_ocr_thread(self, img):
        if self.ocr_thread and self.ocr_thread.isRunning():
            return
        self.ocr_thread = QtCore.QThread()
        self.ocr_worker = OCRWorker(img, do_overlay=True)
        self.ocr_worker.moveToThread(self.ocr_thread)
        self.ocr_thread.started.connect(self.ocr_worker.run)
        self.ocr_worker.finished.connect(self._ocr_finished)
        self.ocr_worker.finished.connect(self.ocr_thread.quit)
        self.ocr_worker.finished.connect(self.ocr_worker.deleteLater)
        self.ocr_thread.finished.connect(self.ocr_thread.deleteLater)
        self.ocr_thread.start()

    def _ocr_finished(self, text, overlay_image):
        # show overlay (scaled)
        if overlay_image is not None:
            self.last_overlay = overlay_image
            pm = cv2_to_qpixmap(overlay_image)
            pm_scaled = pm.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pm_scaled)

        # set text
        self.text_edit.setPlainText(text)
        self.btn_run_ocr.setEnabled(True)
        self.status_label.setText("OCR finished. You can Save Text or run again.")

    def show_last_overlay(self):
        if self.last_overlay is None:
            QtWidgets.QMessageBox.information(self, "Info", "No overlay to show. Run OCR first.")
            return
        pm = cv2_to_qpixmap(self.last_overlay)
        pm_scaled = pm.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pm_scaled)
        self.status_label.setText("Overlay preview shown.")

    def save_text(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.information(self, "Info", "No extracted text to save.")
            return
        filename = os.path.join(SAVE_DIR, f"scanned_text_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        QtWidgets.QMessageBox.information(self, "Saved", f"Extracted text saved to:\n{filename}")
        self.status_label.setText(f"Saved text to {os.path.basename(filename)}")

    def closeEvent(self, ev):
        # stop camera and threads
        try:
            if self.camera_timer.isActive():
                self.camera_timer.stop()
            if self.camera and self.camera.isOpened():
                self.camera.release()
        except Exception:
            pass
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
