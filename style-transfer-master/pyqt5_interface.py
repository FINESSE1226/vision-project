import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
                             QFileDialog, QComboBox, QTextEdit, QProgressBar,
                             QGroupBox, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import torch
from style_transfer_by_torch import StyleTransfer
import time

# Get the base directory (where the script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class StyleTransferThread(QThread):
    """Style transfer processing thread"""
    progress = pyqtSignal(int, str)  # Progress percentage, status info
    finished = pyqtSignal(object, str)  # Result image, status info
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, content_path, style_path, style_weight, content_weight, epochs):
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.epochs = epochs
        self._is_running = True
    
    def run(self):
        try:
            # Create StyleTransfer instance
            st = StyleTransfer(self.content_path, self.style_path, 
                             self.style_weight, self.content_weight)
            
            # Define progress callback
            def progress_callback(current_epoch, total_epochs, loss_value):
                if not self._is_running:
                    return
                progress_percent = int((current_epoch / total_epochs) * 100)
                status = f"Epoch {current_epoch}/{total_epochs}, Loss: {loss_value:.6f}"
                self.progress.emit(progress_percent, status)
            
            # Define stop check function
            def stop_check():
                return not self._is_running
            
            # Call the original main_train method
            combination_param = st.main_train(epoch=self.epochs, 
                                             progress_callback=progress_callback,
                                             stop_check=stop_check)
            
            # Convert the final result to PIL Image
            result_image = st.deprocess_img(combination_param, return_img=True)
            
            self.finished.emit(result_image, "Processing completed!")
            
        except Exception as e:
            import traceback
            error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def stop(self):
        self._is_running = False


class StyleTransferWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.content_image_path = None
        self.style_image_path = None
        self.worker_thread = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Image Style Transfer Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left side: Image selection and parameter adjustment
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)  # Add spacing between groups
        left_layout.setContentsMargins(5, 15, 5, 5)  # Increased top margin for first group
        left_widget.setLayout(left_layout)
        
        # Image selection area
        image_group = QGroupBox("Image Selection")
        image_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 15px; 
                padding-top: 10px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 10px; 
                padding: 0 8px 0 8px; 
                margin-top: 5px;
            }
        """)
        image_layout = QVBoxLayout()
        image_layout.setSpacing(8)
        image_layout.setContentsMargins(10, 25, 10, 10)  # Further increased top margin for title
        
        # Content image selection
        content_group = QGroupBox("Content Image")
        content_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 10px; 
                padding-top: 10px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 10px; 
                padding: 0 8px 0 8px; 
                margin-top: 5px;
            }
        """)
        content_layout = QVBoxLayout()
        content_layout.setSpacing(5)
        content_layout.setContentsMargins(5, 25, 5, 5)  # Further increased top margin for title
        
        self.content_label = QLabel("No image selected")
        self.content_label.setAlignment(Qt.AlignCenter)
        self.content_label.setMinimumHeight(200)
        self.content_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f5f5f5;")
        content_layout.addWidget(self.content_label)
        
        content_btn = QPushButton("Select Content Image")
        content_btn.clicked.connect(self.select_content_image)
        content_layout.addWidget(content_btn)
        
        content_group.setLayout(content_layout)
        image_layout.addWidget(content_group)
        
        # Style image selection
        style_group = QGroupBox("Style Image")
        style_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 10px; 
                padding-top: 10px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 10px; 
                padding: 0 8px 0 8px; 
                margin-top: 5px;
            }
        """)
        style_layout = QVBoxLayout()
        style_layout.setSpacing(5)
        style_layout.setContentsMargins(5, 25, 5, 5)  # Further increased top margin for title
        
        # Preset style selection
        style_layout.addWidget(QLabel("Select Preset Style:"))
        self.preset_style_combo = QComboBox()
        self.preset_style_combo.setEditable(False)
        self.preset_style_combo.setEnabled(True)
        style_layout.addWidget(self.preset_style_combo)
        
        self.style_label = QLabel("No image selected")
        self.style_label.setAlignment(Qt.AlignCenter)
        self.style_label.setMinimumHeight(200)
        self.style_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f5f5f5;")
        style_layout.addWidget(self.style_label)
        
        style_group.setLayout(style_layout)
        image_layout.addWidget(style_group)
        
        # Connect signal after ComboBox is in layout
        self.preset_style_combo.currentTextChanged.connect(self.on_preset_style_changed)
        
        # Load preset styles - use QTimer to ensure UI is fully rendered
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.load_preset_styles)
        
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)
        
        # Parameter adjustment area
        param_group = QGroupBox("⚙️ Parameters")
        param_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 10px; 
                padding-top: 10px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 10px; 
                padding: 0 8px 0 8px; 
                margin-top: 5px;
            }
        """)
        param_layout = QVBoxLayout()
        param_layout.setSpacing(8)
        param_layout.setContentsMargins(10, 25, 10, 10)  # Further increased top margin for title
        
        # Style weight
        style_weight_layout = QHBoxLayout()
        style_weight_layout.addWidget(QLabel("Style Weight:"))
        self.style_weight_slider = QSlider(Qt.Horizontal)
        self.style_weight_slider.setMinimum(1)
        self.style_weight_slider.setMaximum(20)
        self.style_weight_slider.setValue(10)  # Increase default style weight for better effect
        self.style_weight_slider.valueChanged.connect(self.update_style_weight_label)
        style_weight_layout.addWidget(self.style_weight_slider)
        self.style_weight_label = QLabel("10.0")
        self.style_weight_label.setMinimumWidth(50)
        style_weight_layout.addWidget(self.style_weight_label)
        param_layout.addLayout(style_weight_layout)
        
        # Content weight
        content_weight_layout = QHBoxLayout()
        content_weight_layout.addWidget(QLabel("Content Weight:"))
        self.content_weight_slider = QSlider(Qt.Horizontal)
        self.content_weight_slider.setMinimum(1)  # 0.01 * 100
        self.content_weight_slider.setMaximum(10)  # 0.1 * 100
        self.content_weight_slider.setValue(25)  # 0.025 * 100
        self.content_weight_slider.valueChanged.connect(self.update_content_weight_label)
        content_weight_layout.addWidget(self.content_weight_slider)
        self.content_weight_label = QLabel("0.025")
        self.content_weight_label.setMinimumWidth(50)
        content_weight_layout.addWidget(self.content_weight_label)
        param_layout.addLayout(content_weight_layout)
        
        # Training epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Training Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(10)
        self.epochs_spin.setMaximum(300)
        self.epochs_spin.setValue(100)  # Default to 100 epochs like original code
        epochs_layout.addWidget(self.epochs_spin)
        param_layout.addLayout(epochs_layout)
        
        # Process button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        self.process_btn.clicked.connect(self.start_processing)
        param_layout.addWidget(self.process_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop Processing")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        param_layout.addWidget(self.stop_btn)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)
        
        # Progress bar
        progress_group = QGroupBox("Processing Progress")
        progress_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 10px; 
                padding-top: 10px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 10px; 
                padding: 0 8px 0 8px; 
                margin-top: 5px;
            }
        """)
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(5)
        progress_layout.setContentsMargins(10, 25, 10, 10)  # Further increased top margin for title
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        progress_layout.addWidget(self.status_text)
        
        progress_group.setLayout(progress_layout)
        left_layout.addWidget(progress_group)
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # Right side: Result display
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(5, 5, 5, 5)  # Add margins
        right_widget.setLayout(right_layout)
        
        result_group = QGroupBox("Processing Result")
        result_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 10px; 
                padding-top: 10px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 10px; 
                padding: 0 8px 0 8px; 
                margin-top: 5px;
            }
        """)
        result_layout = QVBoxLayout()
        result_layout.setSpacing(8)
        result_layout.setContentsMargins(10, 25, 10, 10)  # Further increased top margin for title
        
        self.result_label = QLabel("Waiting for processing...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(400)
        self.result_label.setStyleSheet("border: 2px solid #ccc; background-color: #fafafa;")
        result_layout.addWidget(self.result_label)
        
        # Save button
        save_btn = QPushButton("Save Result")
        save_btn.clicked.connect(self.save_result)
        result_layout.addWidget(save_btn)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        
        # Set splitter ratio
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
    
    def load_preset_styles(self):
        """Load preset style images"""
        style_dir = os.path.join(BASE_DIR, "images")
        
        if os.path.exists(style_dir):
            for file in sorted(os.listdir(style_dir)):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if file not in ['Taipei101.jpg', 'Run.png']:
                        self.preset_style_combo.addItem(file)
        
        # Only update preview after style_label is created
        if self.preset_style_combo.count() > 0 and hasattr(self, 'style_label'):
            # Set the first item as current to trigger the signal
            self.preset_style_combo.setCurrentIndex(0)
    
    
    def on_preset_style_changed(self, style_name):
        """Preset style changed"""
        if style_name:
            style_path = os.path.join(BASE_DIR, "images", style_name)
            if os.path.exists(style_path):
                self.style_image_path = style_path
                self.display_image(style_path, self.style_label)
    
    def select_content_image(self):
        """Select content image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Content Image", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.content_image_path = file_path
            self.display_image(file_path, self.content_label)
    
    
    def display_image(self, image_path, label):
        """Display image"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale image to fit label
                scaled_pixmap = pixmap.scaled(
                    label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
            else:
                label.setText("Failed to load image")
        except Exception as e:
            label.setText(f"Image loading error: {str(e)}")
    
    def update_style_weight_label(self, value):
        """Update style weight label"""
        self.style_weight_label.setText(f"{value / 10.0:.1f}")
    
    def update_content_weight_label(self, value):
        """Update content weight label"""
        self.content_weight_label.setText(f"{value / 1000.0:.3f}")
    
    def start_processing(self):
        """Start processing"""
        # Check inputs
        if not self.content_image_path:
            QMessageBox.warning(self, "Warning", "Please select a content image!")
            return
        
        if not self.style_image_path:
            QMessageBox.warning(self, "Warning", "Please select a style image!")
            return
        
        # Get parameters
        style_weight = self.style_weight_slider.value() / 10.0
        content_weight = self.content_weight_slider.value() / 1000.0
        epochs = self.epochs_spin.value()
        
        # Update UI state
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.result_label.clear()
        self.result_label.setText("Processing...")
        
        # Create and start worker thread
        self.worker_thread = StyleTransferThread(
            self.content_image_path, self.style_image_path,
            style_weight, content_weight, epochs
        )
        self.worker_thread.progress.connect(self.update_progress)
        self.worker_thread.finished.connect(self.on_processing_finished)
        self.worker_thread.error.connect(self.on_processing_error)
        self.worker_thread.start()
    
    def stop_processing(self):
        """Stop processing"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.status_text.append("Processing stopped")
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def update_progress(self, percent, status):
        """Update progress"""
        self.progress_bar.setValue(percent)
        self.status_text.append(status)
        # Auto scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_processing_finished(self, result_image, status):
        """Processing finished"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_text.append(status)
        self.progress_bar.setValue(100)
        
        # Save result image
        self.result_image = result_image
        
        # Display result
        if result_image:
            try:
                # Convert to RGB mode
                if result_image.mode != 'RGB':
                    result_image = result_image.convert('RGB')
                
                # Convert to QPixmap
                bytes_img = result_image.tobytes("raw", "RGB")
                q_image = QImage(bytes_img, result_image.size[0], result_image.size[1], 
                               QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # Scale to fit label
                scaled_pixmap = pixmap.scaled(
                    self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.result_label.setPixmap(scaled_pixmap)
            except Exception as e:
                self.status_text.append(f"Error displaying result: {str(e)}")
    
    def on_processing_error(self, error_msg):
        """Handle processing error"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_text.append(f" {error_msg}")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error_msg}")
        self.result_label.clear()
        self.result_label.setText("Processing failed")
    
    def save_result(self):
        """Save result"""
        if not hasattr(self, 'result_image') or self.result_image is None:
            QMessageBox.warning(self, "Warning", "No result to save!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", "", 
            "PNG Image (*.png);;JPEG Image (*.jpg)")
        if file_path:
            try:
                self.result_image.save(file_path)
                QMessageBox.information(self, "Success", "Image saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = StyleTransferWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

