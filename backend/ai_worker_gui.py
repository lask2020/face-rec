import sys
import os
import json
import logging
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QFrame, QGridLayout,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QThread
from PyQt6.QtGui import QFont, QTextCursor

# Ensure the parent directories are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LogEmitter(QObject):
    """Signal emitter for routing standard logging to the PyQt GUI safely."""
    log_signal = pyqtSignal(str)


class PyQtLogHandler(logging.Handler):
    """Logging handler that emits log messages via PyQt signals."""
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_signal.emit(msg)


class WorkerThread(QThread):
    """Thread wrapper to run the grpc client as a subprocess without blocking the GUI."""
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, url, provider, plate_settings=None):
        super().__init__()
        self.url = url
        self.provider = provider
        self.plate_settings = plate_settings or {}
        self.process = None
        self.is_running = True

    def run(self):
        import subprocess
        try:
            # Run ai_worker_grpc.py as a separate process to avoid CoreML + PyQt thread crashes
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, "--run-worker"]
            else:
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_worker_grpc.py")
                cmd = [sys.executable, script_path]

            env = os.environ.copy()
            env["CONTROL_PLANE_URL"] = self.url
            env["ONNX_PROVIDER"] = self.provider
            # Plate-tracking tunables — worker reads these from env (see ai_worker_grpc.py)
            for key, value in self.plate_settings.items():
                env[key] = str(value)

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )

            # Read logs line by line
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                if line:
                    self.log_signal.emit(line.strip())
            
            self.process.stdout.close()
            self.process.wait()

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


class AIWorkerWindow(QMainWindow):
    CONFIG_FILE = "worker_config.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceRec AI Worker Node")
        self.setMinimumSize(700, 500)

        self.worker_thread = None
        self.is_running = False

        self.url_value = os.getenv("CONTROL_PLANE_URL", "localhost:50051")
        self.provider_value = "CPUExecutionProvider"
        # Plate-tracking defaults (mirror ai_worker_grpc.py)
        self.min_plate_hits_value = 1
        self.plate_track_timeout_value = 6.0
        self.plate_track_max_duration_value = 12.0
        self._load_config()

        self.setup_ui()
        self.setup_logging()

    def _load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.url_value = config.get("url", self.url_value)
                    self.provider_value = config.get("provider", self.provider_value)
                    self.min_plate_hits_value = config.get("min_plate_hits", self.min_plate_hits_value)
                    self.plate_track_timeout_value = config.get("plate_track_timeout", self.plate_track_timeout_value)
                    self.plate_track_max_duration_value = config.get("plate_track_max_duration", self.plate_track_max_duration_value)
            except Exception as e:
                logging.error(f"Failed to load config: {e}")

    def _save_config(self):
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump({
                    "url": self.url_entry.text().strip(),
                    "provider": self.provider_combo.currentText(),
                    "min_plate_hits": self.min_plate_hits_spin.value(),
                    "plate_track_timeout": self.plate_track_timeout_spin.value(),
                    "plate_track_max_duration": self.plate_track_max_duration_spin.value(),
                }, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ── Title ──
        title_lbl = QLabel("⚙️ Configuration")
        title_font = QFont("Helvetica", 16, QFont.Weight.Bold)
        title_lbl.setFont(title_font)
        main_layout.addWidget(title_lbl)

        # ── Config Form ──
        form_layout = QGridLayout()
        form_layout.setColumnStretch(1, 1)

        # Control Plane URL
        url_lbl = QLabel("Control Plane URL:")
        url_lbl.setFont(QFont("Helvetica", 13))
        self.url_entry = QLineEdit(self.url_value)
        self.url_entry.setFont(QFont("Helvetica", 13))
        self.url_entry.editingFinished.connect(self._save_config)
        
        # Start/Stop Button
        self.start_btn = QPushButton("⚡ Start Worker")
        self.start_btn.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        self.start_btn.setMinimumHeight(45)
        self.start_btn.clicked.connect(self.toggle_worker)
        # Add PyQt styling for the button
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1f538d;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #14375d;
            }
        """)

        form_layout.addWidget(url_lbl, 0, 0)
        form_layout.addWidget(self.url_entry, 0, 1)
        form_layout.addWidget(self.start_btn, 0, 2, 2, 1) # Span 2 rows

        # Execution Provider
        provider_lbl = QLabel("Execution Provider:")
        provider_lbl.setFont(QFont("Helvetica", 13))
        self.provider_combo = QComboBox()
        self.provider_combo.setFont(QFont("Helvetica", 13))
        providers = [
            "CPUExecutionProvider",
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "ROCmExecutionProvider",
            "DmlExecutionProvider"
        ]
        self.provider_combo.addItems(providers)
        
        # Set selected provider
        index = self.provider_combo.findText(self.provider_value)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)
            
        self.provider_combo.currentTextChanged.connect(self._save_config)

        form_layout.addWidget(provider_lbl, 1, 0)
        form_layout.addWidget(self.provider_combo, 1, 1)

        main_layout.addLayout(form_layout)

        # ── Plate Detection Tuning ──
        plate_title = QLabel("🚗 Plate Detection")
        plate_title.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        main_layout.addWidget(plate_title)

        plate_form = QGridLayout()
        plate_form.setColumnStretch(1, 1)
        plate_form.setColumnStretch(3, 1)

        # Min Plate Hits
        hits_lbl = QLabel("Min Hits:")
        hits_lbl.setFont(QFont("Helvetica", 13))
        hits_lbl.setToolTip("Discard plate tracks seen in fewer frames than this. "
                            "Lower = catch fast-passing plates, higher = fewer false positives.")
        self.min_plate_hits_spin = QSpinBox()
        self.min_plate_hits_spin.setFont(QFont("Helvetica", 13))
        self.min_plate_hits_spin.setRange(1, 20)
        self.min_plate_hits_spin.setValue(int(self.min_plate_hits_value))
        self.min_plate_hits_spin.valueChanged.connect(self._save_config)

        # Track Timeout
        timeout_lbl = QLabel("Track Timeout (s):")
        timeout_lbl.setFont(QFont("Helvetica", 13))
        timeout_lbl.setToolTip("Seconds of inactivity before a plate track flushes. "
                               "Wider window lets sparse detections of the same plate accumulate hits.")
        self.plate_track_timeout_spin = QDoubleSpinBox()
        self.plate_track_timeout_spin.setFont(QFont("Helvetica", 13))
        self.plate_track_timeout_spin.setRange(0.5, 60.0)
        self.plate_track_timeout_spin.setSingleStep(0.5)
        self.plate_track_timeout_spin.setValue(float(self.plate_track_timeout_value))
        self.plate_track_timeout_spin.valueChanged.connect(self._save_config)

        # Max Duration
        maxdur_lbl = QLabel("Max Duration (s):")
        maxdur_lbl.setFont(QFont("Helvetica", 13))
        maxdur_lbl.setToolTip("Maximum lifetime of a plate track before it is force-flushed.")
        self.plate_track_max_duration_spin = QDoubleSpinBox()
        self.plate_track_max_duration_spin.setFont(QFont("Helvetica", 13))
        self.plate_track_max_duration_spin.setRange(1.0, 120.0)
        self.plate_track_max_duration_spin.setSingleStep(1.0)
        self.plate_track_max_duration_spin.setValue(float(self.plate_track_max_duration_value))
        self.plate_track_max_duration_spin.valueChanged.connect(self._save_config)

        plate_form.addWidget(hits_lbl, 0, 0)
        plate_form.addWidget(self.min_plate_hits_spin, 0, 1)
        plate_form.addWidget(timeout_lbl, 0, 2)
        plate_form.addWidget(self.plate_track_timeout_spin, 0, 3)
        plate_form.addWidget(maxdur_lbl, 1, 0)
        plate_form.addWidget(self.plate_track_max_duration_spin, 1, 1)

        main_layout.addLayout(plate_form)

        # ── Separator ──
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # ── Application Logs ──
        log_lbl = QLabel("📋 Application Logs")
        log_lbl.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        main_layout.addWidget(log_lbl)

        self.log_textedit = QTextEdit()
        self.log_textedit.setReadOnly(True)
        self.log_textedit.setFont(QFont("Courier", 12))
        self.log_textedit.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        main_layout.addWidget(self.log_textedit)

    def setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = [] # Clear old handlers

        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.append_log)

        # GUI Handler
        gui_handler = PyQtLogHandler(self.log_emitter)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        gui_handler.setFormatter(formatter)
        root_logger.addHandler(gui_handler)

        # Terminal Handler
        term_handler = logging.StreamHandler(sys.stdout)
        term_handler.setFormatter(formatter)
        root_logger.addHandler(term_handler)

        logging.info("PyQt6 GUI Initialized. Ready to rock!")

    def append_log(self, text):
        self.log_textedit.moveCursor(QTextCursor.MoveOperation.End)
        self.log_textedit.insertPlainText(text + "\n")
        self.log_textedit.moveCursor(QTextCursor.MoveOperation.End)

    def toggle_worker(self):
        if self.is_running:
            self.stop_worker()
        else:
            self.start_worker()

    def start_worker(self):
        url = self.url_entry.text().strip()
        provider = self.provider_combo.currentText()

        if not url:
            logging.error("Control Plane URL cannot be empty.")
            return

        plate_settings = {
            "MIN_PLATE_HITS": self.min_plate_hits_spin.value(),
            "PLATE_TRACK_TIMEOUT": self.plate_track_timeout_spin.value(),
            "PLATE_TRACK_MAX_DURATION": self.plate_track_max_duration_spin.value(),
        }

        self.url_entry.setEnabled(False)
        self.provider_combo.setEnabled(False)
        self.min_plate_hits_spin.setEnabled(False)
        self.plate_track_timeout_spin.setEnabled(False)
        self.plate_track_max_duration_spin.setEnabled(False)
        self.start_btn.setText("🛑 Stop Worker")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #a83232;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #7a2424;
            }
        """)

        self.is_running = True
        
        self.worker_thread = WorkerThread(url, provider, plate_settings)
        self.worker_thread.finished_signal.connect(self.on_worker_finished)
        self.worker_thread.error_signal.connect(self.on_worker_error)
        self.worker_thread.log_signal.connect(self.append_log)
        self.worker_thread.start()
        
        logging.info(f"Starting worker thread connecting to {url} with {provider}...")

    def stop_worker(self):
        if self.worker_thread and self.worker_thread.isRunning():
            logging.info("Sending stop signal to worker thread...")
            self.worker_thread.stop()

    def on_worker_finished(self):
        self.is_running = False
        self.start_btn.setText("⚡ Start Worker")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1f538d;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #14375d;
            }
        """)
        self.url_entry.setEnabled(True)
        self.provider_combo.setEnabled(True)
        self.min_plate_hits_spin.setEnabled(True)
        self.plate_track_timeout_spin.setEnabled(True)
        self.plate_track_max_duration_spin.setEnabled(True)
        logging.info("Worker thread completely stopped.")

    def on_worker_error(self, err_msg):
        logging.error(f"Worker crashed: {err_msg}")


def main():
    # Fix High DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Native-like modern look
    
    window = AIWorkerWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-worker":
        import ai_worker_grpc
        ai_worker_grpc.run_grpc_client()
        sys.exit(0)

    if len(sys.argv) > 1 and ("--help" in sys.argv or "-h" in sys.argv):
        print("Usage: python ai_worker_gui.py")
        sys.exit(0)

    main()
