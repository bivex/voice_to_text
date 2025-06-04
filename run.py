import sys
import torch
import sounddevice as sd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QLabel, QStatusBar, QAction, QMessageBox,
    QFrame, QSizePolicy, QProgressBar, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPoint
from PyQt5.QtGui import QIcon, QFont, QKeySequence, QPalette, QColor, QPixmap, QPainter, QBrush, QPen
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from collections import deque
import threading
import queue
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLORS = {
    'background': '#F5F5F5',  # Light gray background
    'text': '#202020',        # Dark gray text
    'primary': '#0078D4',     # Windows blue
    'secondary': '#2B88D8',   # Lighter blue
    'danger': '#D83B01',      # Windows red
    'success': '#107C10',     # Windows green
    'border': '#E0E0E0',      # Light border
    'focus': '#0078D4',       # Windows blue
    'disabled': '#CCCCCC',    # Light gray
    'hover': '#E5F1FB',       # Light blue hover
    'pressed': '#C7E0F4',     # Darker blue pressed
    'header': '#0078D4',      # Windows blue for header
    'header_text': '#FFFFFF', # White text for header
    'status': '#F0F0F0'       # Light gray for status bar
}

class AccessibleButton(QPushButton):
    def __init__(self, text, shortcut=None, parent=None):
        super().__init__(text, parent)
        if shortcut:
            self.setShortcut(shortcut)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(44)  # Minimum touch target size
        self.setMinimumWidth(100)
        
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.setStyleSheet(self.styleSheet() + f"""
            QPushButton:focus {{
                border: 3px solid {COLORS['focus']};
                outline: none;
            }}
        """)

# Singleton for Audio Device Management
class AudioDeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioDeviceManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.devices = sd.query_devices()
        logger.info("Audio devices initialized")

    def get_devices(self) -> List[Dict[str, Any]]:
        return self.devices

    def print_devices(self):
        logger.info("\nAvailable Audio Devices:")
        for i, device in enumerate(self.devices):
            logger.info(f"Device {i}:")
            logger.info(f"  Name: {device['name']}")
            logger.info(f"  Input Channels: {device['max_input_channels']}")
            logger.info(f"  Output Channels: {device['max_output_channels']}")
            logger.info(f"  Default Sample Rate: {device['default_samplerate']}")

# Observer Pattern for Audio Events
class AudioObserver(ABC):
    @abstractmethod
    def update(self, event_type: str, data: Any = None):
        pass

# Command Pattern for Actions
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

class RecordCommand(Command):
    def __init__(self, recorder):
        self.recorder = recorder

    def execute(self):
        self.recorder.start_recording()

    def undo(self):
        self.recorder.stop_recording()

# Factory Pattern for UI Components
class UIComponentFactory:
    @staticmethod
    def create_button(text: str, shortcut: str = None, parent: QWidget = None) -> QPushButton:
        button = AccessibleButton(text, shortcut, parent)
        return button

    @staticmethod
    def create_progress_bar(parent: QWidget = None) -> QProgressBar:
        progress_bar = QProgressBar(parent)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p%")
        return progress_bar

# Model
class AudioRecorderModel:
    def __init__(self, processor=None, model=None):
        self.sample_rate = 16000
        self.chunk_duration = 5  # Duration of each chunk for processing
        self.buffer_size = 10  # Number of chunks to keep in memory
        self.observers: List[AudioObserver] = []
        self.recording = False
        self.paused = False
        self.current_time = 0
        self.processor = processor
        self.model = model
        self.accumulated_text = ""
        self.audio_buffer = deque(maxlen=self.buffer_size)  # Circular buffer for audio chunks
        self.processing_queue = queue.Queue()  # Queue for processing chunks
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def add_audio_chunk(self, chunk):
        with self.lock:
            self.audio_buffer.append(chunk)
            self.processing_queue.put(chunk)

    def get_next_chunk(self):
        try:
            return self.processing_queue.get_nowait()
        except queue.Empty:
            return None

    def add_observer(self, observer: AudioObserver):
        self.observers.append(observer)

    def remove_observer(self, observer: AudioObserver):
        self.observers.remove(observer)

    def notify_observers(self, event_type: str, data: Any = None):
        for observer in self.observers:
            observer.update(event_type, data)

def generate_random_icon(size=64):
    """Generate a random icon with a unique pattern."""
    # Create a pixmap with the specified size
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # Create a painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Generate random colors
    colors = [
        QColor(0, 120, 212),  # Windows blue
        QColor(0, 153, 204),  # Light blue
        QColor(0, 99, 177),   # Dark blue
        QColor(0, 120, 215),  # Accent blue
    ]
    
    # Draw random shapes
    for _ in range(3):
        # Random shape type (0: circle, 1: square, 2: triangle)
        shape_type = random.randint(0, 2)
        color = random.choice(colors)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(), 2))
        
        # Random position and size
        x = random.randint(0, size-20)
        y = random.randint(0, size-20)
        w = random.randint(20, size-x)
        h = random.randint(20, size-y)
        
        if shape_type == 0:  # Circle
            painter.drawEllipse(x, y, w, h)
        elif shape_type == 1:  # Square
            painter.drawRect(x, y, w, h)
        else:  # Triangle
            points = [
                (x + w//2, y),
                (x, y + h),
                (x + w, y + h)
            ]
            painter.drawPolygon(*[QPoint(p[0], p[1]) for p in points])
    
    painter.end()
    return pixmap

# View
class MainView(QMainWindow):
    def __init__(self, model: AudioRecorderModel):
        super().__init__()
        self.model = model
        self.controller = None
        self.initUI()
        
        # Set random icon
        self.setWindowIcon(QIcon(generate_random_icon()))
        
        # Change icon every 5 minutes
        self.icon_timer = QTimer(self)
        self.icon_timer.timeout.connect(self.update_icon)
        self.icon_timer.start(300000)  # 300000 ms = 5 minutes
        
    def set_controller(self, controller):
        self.controller = controller

    def on_recording_finished(self, text):
        self.text_output.append(text)
        self.reset_buttons()
        
    def on_chunk_processed(self, text):
        if text.strip():
            self.text_output.append(text)
            # Auto-scroll to bottom
            self.text_output.verticalScrollBar().setValue(
                self.text_output.verticalScrollBar().maximum()
            )
        
    def update_status(self, message):
        self.statusBar.showMessage(message)
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def start_recording(self):
        if not self.recorder.model.recording:
            self.record_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.model.accumulated_text = ""  # Reset accumulated text
            self.recorder.start()
            
    def toggle_pause(self):
        if self.recorder.model.recording:
            self.recorder.model.paused = not self.recorder.model.paused
            self.pause_button.setText("⏸️ Resume (P)" if self.recorder.model.paused else "⏸️ Pause (P)")
            self.statusBar.showMessage("Paused" if self.recorder.model.paused else "Recording...")
            
    def stop_recording(self):
        if self.recorder.model.recording:
            self.recorder.model.recording = False
            self.recorder.wait()
            self.reset_buttons()
            self.progress_bar.setValue(0)  # Reset progress bar when stopped
            
    def reset_buttons(self):
        self.record_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("⏸️ Pause (P)")
        self.progress_bar.setValue(0)
        self.record_button.setFocus()

    def copy_text(self):
        text = self.text_output.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.statusBar.showMessage("Text copied to clipboard", 3000)
        else:
            self.statusBar.showMessage("No text to copy", 3000)
            
    def clear_text(self):
        reply = QMessageBox.question(
            self, 'Confirm Clear',
            'Are you sure you want to clear all text?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.text_output.clear()
            self.statusBar.showMessage("Text cleared", 3000)
        
    def close_application(self):
        if self.recorder.model.recording:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Recording is in progress. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.recorder.model.recording = False
                self.recorder.wait()
                QApplication.quit()
        else:
            QApplication.quit()
            
    def closeEvent(self, event):
        self.close_application()
        event.accept()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("Russian Speech Recognition")
        self.setMinimumSize(900, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Create header with modern styling
        header_widget = QWidget()
        header_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['header']};
                border-radius: 4px;
            }}
        """)
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create title label with modern font
        title_label = QLabel("Russian Speech Recognition")
        title_font = QFont("Segoe UI", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['header_text']};
                background-color: transparent;
            }}
        """)
        header_layout.addWidget(title_label)
        main_layout.addWidget(header_widget)
        
        # Create main content area
        content_widget = QWidget()
        content_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['background']};
                border-radius: 4px;
            }}
        """)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create text output area with modern styling
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMinimumHeight(400)
        self.text_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: white;
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 12px;
                font-family: 'Segoe UI';
                font-size: 12pt;
                line-height: 1.5;
            }}
            QTextEdit:focus {{
                border: 2px solid {COLORS['focus']};
                outline: none;
            }}
        """)
        content_layout.addWidget(self.text_output)
        
        # Create progress bar with modern styling
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                text-align: center;
                height: 24px;
                font-family: 'Segoe UI';
                font-size: 11pt;
                color: {COLORS['text']};
                background-color: white;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """)
        content_layout.addWidget(self.progress_bar)
        
        # Create button containers with modern styling
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(12)
        
        text_container = QWidget()
        text_layout = QHBoxLayout(text_container)
        text_layout.setSpacing(12)
        
        # Create control buttons with modern styling
        self.record_button = AccessibleButton("Record (R)", "R")
        self.pause_button = AccessibleButton("Pause (P)", "P")
        self.stop_button = AccessibleButton("Stop (S)", "S")
        
        # Create text operation buttons
        self.copy_button = AccessibleButton("Copy Text (Ctrl+C)", "Ctrl+C")
        self.clear_button = AccessibleButton("Clear Text (Ctrl+L)", "Ctrl+L")
        self.exit_button = AccessibleButton("Exit (Esc)", "Esc")
        
        # Modern button styling
        button_style = f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: 'Segoe UI';
                font-size: 11pt;
                font-weight: bold;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['pressed']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['disabled']};
                color: {COLORS['text']};
            }}
        """
        
        # Apply styles to buttons
        self.record_button.setStyleSheet(button_style)
        self.pause_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
        self.copy_button.setStyleSheet(button_style)
        self.clear_button.setStyleSheet(button_style)
        self.exit_button.setStyleSheet(button_style)
        
        # Initially disable pause and stop buttons
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Add control buttons to control layout
        control_layout.addWidget(self.record_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        
        # Add text operation buttons to text layout
        text_layout.addWidget(self.copy_button)
        text_layout.addWidget(self.clear_button)
        text_layout.addWidget(self.exit_button)
        
        # Add button containers to content layout
        content_layout.addWidget(control_container)
        content_layout.addWidget(text_container)
        
        # Add content widget to main layout
        main_layout.addWidget(content_widget)
        
        # Create status bar with modern styling
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['status']};
                color: {COLORS['text']};
                font-family: 'Segoe UI';
                font-size: 10pt;
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        self.statusBar.showMessage("Ready")
        
        # Initialize audio recorder
        self.recorder = AudioRecorderThread(self.model)
        self.recorder.finished.connect(self.on_recording_finished)
        self.recorder.status_update.connect(self.update_status)
        self.recorder.progress_update.connect(self.update_progress)
        self.recorder.chunk_processed.connect(self.on_chunk_processed)
        
        # Connect button clicks
        self.record_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_recording)
        self.copy_button.clicked.connect(self.copy_text)
        self.clear_button.clicked.connect(self.clear_text)
        self.exit_button.clicked.connect(self.close_application)
        
        # Set focus to first button
        self.record_button.setFocus()

    def update_icon(self):
        """Update the window icon with a new random icon."""
        self.setWindowIcon(QIcon(generate_random_icon()))

# Controller
class MainController:
    def __init__(self, model: AudioRecorderModel, view: MainView):
        self.model = model
        self.view = view
        self.command_history = []

    def execute_command(self, command: Command):
        command.execute()
        self.command_history.append(command)

    def undo_last_command(self):
        if self.command_history:
            command = self.command_history.pop()
            command.undo()

# Audio Recorder Thread
class AudioRecorderThread(QThread):
    finished = pyqtSignal(str)
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    chunk_processed = pyqtSignal(str)

    def __init__(self, model: AudioRecorderModel):
        super().__init__()
        self.model = model
        self.processing_thread = None
        self.stop_processing = False

    def run(self):
        try:
            self.model.recording = True
            self.model.paused = False
            self.model.current_time = 0
            self.status_update.emit("Recording in progress...")

            # Start processing thread
            self.stop_processing = False
            self.processing_thread = threading.Thread(target=self.process_audio_chunks)
            self.processing_thread.start()

            chunk_size = int(self.model.sample_rate * self.model.chunk_duration)
            
            while self.model.recording:
                if self.model.paused:
                    self.msleep(100)
                    continue

                # Record a chunk
                audio_data = sd.rec(chunk_size,
                                  samplerate=self.model.sample_rate,
                                  channels=1,
                                  dtype='float32')
                sd.wait()

                if not self.model.recording:
                    break

                # Add chunk to buffer
                self.model.add_audio_chunk(audio_data)
                self.model.current_time += self.model.chunk_duration
                progress = int((self.model.current_time % 100) / 100 * 100)
                self.progress_update.emit(progress)

            # Stop processing thread
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join()

            self.status_update.emit("Recording stopped")
            self.finished.emit(self.model.accumulated_text.strip())

        except Exception as e:
            logger.error(f"Error in audio recording: {str(e)}")
            self.status_update.emit(f"Error: {str(e)}")
            self.model.recording = False

    def process_audio_chunks(self):
        while not self.stop_processing:
            chunk = self.model.get_next_chunk()
            if chunk is None:
                self.msleep(10)  # Small delay to prevent CPU overuse
                continue

            try:
                # Process the chunk
                if len(chunk.shape) > 1:
                    chunk = chunk.mean(axis=1)

                chunk = chunk / np.max(np.abs(chunk))
                audio_tensor = torch.FloatTensor(chunk)

                inputs = self.model.processor(audio_tensor, 
                                           sampling_rate=self.model.sample_rate, 
                                           return_tensors="pt", 
                                           padding=True)

                with torch.no_grad():
                    logits = self.model.model(inputs.input_values, 
                                           attention_mask=inputs.attention_mask).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_sentence = self.model.processor.batch_decode(predicted_ids)[0]

                if predicted_sentence.strip():
                    self.chunk_processed.emit(predicted_sentence)
                    with self.model.lock:
                        self.model.accumulated_text += predicted_sentence + " "

            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

# Application Entry Point
def main():
    # Initialize application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set high contrast palette
    palette = QPalette()
    for role, color in {
        QPalette.Window: COLORS['background'],
        QPalette.WindowText: COLORS['text'],
        QPalette.Base: COLORS['background'],
        QPalette.AlternateBase: COLORS['background'],
        QPalette.ToolTipBase: COLORS['background'],
        QPalette.ToolTipText: COLORS['text'],
        QPalette.Text: COLORS['text'],
        QPalette.Button: COLORS['background'],
        QPalette.ButtonText: COLORS['text'],
        QPalette.BrightText: COLORS['text'],
        QPalette.Link: COLORS['primary'],
        QPalette.Highlight: COLORS['primary'],
        QPalette.HighlightedText: COLORS['background']
    }.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)

    # Initialize audio device manager
    audio_manager = AudioDeviceManager()
    audio_manager.print_devices()

    # Load model
    logger.info("Loading model and processor...")
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    device = torch.device("cpu")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully on CPU!")

    # Initialize MVC components
    audio_model = AudioRecorderModel(processor=processor, model=model)
    main_view = MainView(audio_model)  # Pass model directly to view
    main_controller = MainController(audio_model, main_view)
    main_view.set_controller(main_controller)  # Set controller after creation

    # Show the application
    main_view.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 