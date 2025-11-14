import os
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (   QApplication, QWidget, QLabel,
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QRadioButton, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QFileDialog, QGraphicsDropShadowEffect) 
from PyQt5.QtGui import QPixmap, QImage, QPainter, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import requests
import warnings


# 这里可以根据需要修改为实际的后端服务地址
Server_URL = "http://localhost:5000"  # 后端服务地址


#  创建 image_label——带阴影
def create_image_label():
    label = QLabel()
    label.setFixedSize(640, 640)
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("border: 1px solid #ccc; border-radius: 8px;")
    
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(10)
    shadow.setColor(Qt.black)
    shadow.setOffset(0, 2)
    label.setGraphicsEffect(shadow)
    
    return label

# 忽略所有 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 或者更精确地过滤 sipPyTypeDict 相关警告
warnings.filterwarnings("ignore", message="sipPyTypeDict.*is deprecated")

class PredictionThread(QThread):
    finished_signal = pyqtSignal(dict)  # 成功返回结果
    error_signal = pyqtSignal(str)     # 出现错误

    def __init__(self, file_path,parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        try:
            with open(self.file_path, 'rb') as f:
                files = {'file': (os.path.basename(self.file_path), f)}
                # 获取当前 UI 中的阈值
                conf_threshold = self.parent().confidence_threshold_spinbox.value() / 100  # 转换为 0~1
                iou_threshold = self.parent().iou_threshold_spinbox.value() / 100

                # 发送 POST 请求并带上阈值
                data = {
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold
                }
                response = requests.post(Server_URL + '/predict', files=files, data=data,timeout=60*5)# 超时设置5min
                if response.status_code == 200:
                    try:
                        self.finished_signal.emit(response.json())  # 只有成功解析 JSON 才 emit
                    except ValueError:
                        self.error_signal.emit("返回数据不是有效的 JSON")
                else:
                    self.error_signal.emit(f"服务器返回错误码：{response.status_code}")
        except requests.exceptions.ConnectionError:
            self.error_signal.emit("连接失败：后端服务未运行")
        except Exception as e:
            self.error_signal.emit(f"发生异常：{str(e)}")
            

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("脑肿瘤智能检测与诊断系统")
        self.resize(1000, 850)
        self.init_ui()
        self.setup_ui()
        self.conf_threshold=0.3 # 置信度阈值
        self.iou_threshold=0.7# IoU阈值
        
        self.setStyleSheet("""
            QWidget {
                font-family: "微软雅黑", "Arial", sans-serif;
                font-size: 14px;
                background-color: #f9f9f9;
            }

            QLabel {
                color: #333;
            }

            QPushButton {
                background-color: #007BFF;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                color: white;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #0056b3;
            }

            QPushButton:pressed {
                background-color: #003f7f;
            }

            QSpinBox, QComboBox {
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }

            QTableWidget {
                border: 1px solid #ddd;
                gridline-color: #eee;
            }

            QTableWidget::item:selected {
                background-color: #d0e7ff;
            }

            QCheckBox, QRadioButton {
                spacing: 5px;
            }

            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }

            QGroupBox {
                border: 1px solid #ccc;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }

            QLabel#recognition_result_text {
                background-color: #fff;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 6px;
            }

            QMessageBox {
                background-color: #fff;
            }
        """)
        
        

    def init_ui(self):
        # 顶部区域
        self.total_targets_label = QLabel("总目标数:")
        self.time_label = QLabel("用时:")
        self.target_selection_label = QLabel("目标选择:")
        self.target_selection_combo = QComboBox()

        # 窗口选择区域
        self.window1_label = QLabel("窗口1:")
        self.segmentation_result_checkbox = QtWidgets.QCheckBox("显示分割结果")
        self.detection_box_checkbox = QtWidgets.QCheckBox("显示检测框与标签")
        self.window2_label = QLabel("窗口2:")
        self.mask_radio = QRadioButton("显示Mask")
        self.mask_radio.setChecked(True)  # 默认显示 mask
        self.original_image_radio = QRadioButton("显示原始图片")

        # 阈值区域
        self.confidence_threshold_label = QLabel("置信度阈值:")
        self.confidence_threshold_spinbox = QSpinBox()
        self.confidence_threshold_spinbox.setRange(0, 100)
        self.confidence_threshold_spinbox.setValue(50)
        self.confidence_threshold_spinbox.setSuffix("%")

        self.iou_threshold_label = QLabel("交并比阈值:")
        self.iou_threshold_spinbox = QSpinBox()
        self.iou_threshold_spinbox.setRange(0, 100)
        self.iou_threshold_spinbox.setValue(70)
        self.iou_threshold_spinbox.setSuffix("%")

        # 识别结果区域
        self.recognition_result_label = QLabel("识别结果:")
        self.recognition_result_text = QLabel("暂无")
        self.recognition_result_text.setWordWrap(True)
        self.recognition_result_text.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #dddddd;
            padding: 10px;
            border-radius: 6px;
            min-height: 60px;
        """)
        self.confidence_label = QLabel("置信度:")
        self.result_text = QLabel("暂无")

        # 目标位置区域
        self.target_location_label = QLabel("目标位置:")
        self.xmin_label = QLabel("xmin:")
        self.ymin_label = QLabel("ymin:")
        self.xmax_label = QLabel("xmax:")
        self.ymax_label = QLabel("ymax:")

        # 表格区域
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号", "中文类别","英文类别", "置信度", "坐标位置"])
        self.table.verticalHeader().setVisible(False) # 隐藏行号

        # 按钮区域
        self.open_image_button = QPushButton("打开图片/.dcm影像")
        self.save_result_button = QPushButton("保存结果")
        self.exit_button = QPushButton("退出系统")

        # 图像显示区域
        self.image_label1 = create_image_label()
        self.image_label1.setFixedSize(512, 512)  # 设置固定大小
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setStyleSheet("border: 1px solid black;")  # 添加边框

        self.image_label2 = create_image_label()
        self.image_label2.setFixedSize(512, 512)  # 设置固定大小
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setStyleSheet("border: 1px solid black;")  # 添加边框

        # 刷新按钮
        self.refresh_window1_button = QPushButton("刷新")
        self.refresh_window1_button.setFixedSize(60, 30)

    def setup_ui(self):
        # 布局
        main_layout = QVBoxLayout()

        # 顶部布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.total_targets_label)
        top_layout.addWidget(self.time_label)
        top_layout.addWidget(self.target_selection_label)
        top_layout.addWidget(self.target_selection_combo)
        main_layout.addLayout(top_layout)

        # 窗口选择布局
        window_layout = QHBoxLayout()
        window_layout.addWidget(self.window1_label)
        window_layout.addWidget(self.segmentation_result_checkbox)
        window_layout.addWidget(self.detection_box_checkbox)
        window_layout.addWidget(self.window2_label)
        window_layout.addWidget(self.mask_radio)
        window_layout.addWidget(self.original_image_radio)
        main_layout.addLayout(window_layout)

        # 阈值布局
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.confidence_threshold_label)
        threshold_layout.addWidget(self.confidence_threshold_spinbox)
        threshold_layout.addWidget(self.iou_threshold_label)
        threshold_layout.addWidget(self.iou_threshold_spinbox)
        main_layout.addLayout(threshold_layout)

        # 识别结果布局
        result_layout = QHBoxLayout()
        result_layout.addWidget(self.recognition_result_label)
        result_layout.addWidget(self.recognition_result_text)
        result_layout.addWidget(self.confidence_label)
        result_layout.addWidget(self.result_text)
        main_layout.addLayout(result_layout)

        # 目标位置布局
        location_layout = QGridLayout()
        location_layout.addWidget(self.target_location_label, 0, 0, 1, 2)
        location_layout.addWidget(self.xmin_label, 1, 0)
        location_layout.addWidget(self.ymin_label, 1, 1)
        location_layout.addWidget(self.xmax_label, 2, 0)
        location_layout.addWidget(self.ymax_label, 2, 1)
        main_layout.addLayout(location_layout)

        # 图像显示布局
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)

        # 创建一个容器用于放置图片和刷新按钮
        self.image_display_container = QWidget()
        self.image_display_layout = QVBoxLayout(self.image_display_container)
        self.image_display_layout.setContentsMargins(0, 0, 0, 0)
        self.image_display_layout.setAlignment(Qt.AlignCenter)
        self.image_display_layout.addWidget(self.image_label1)

        # 添加刷新按钮并设置为浮动在左下角
        self.refresh_window1_button.setParent(self.image_display_container)
        self.refresh_window1_button.raise_()
        self.refresh_window1_button.move(5, self.image_label1.height() - self.refresh_window1_button.height() - 5)

        # 主图像布局
        main_image_layout = QHBoxLayout()
        main_image_layout.addWidget(self.image_display_container)
        main_image_layout.addWidget(self.image_label2)

        main_layout.addLayout(main_image_layout)

        # 表格布局
        
        main_layout.addWidget(self.table)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_image_button)
        button_layout.addWidget(self.save_result_button)
        button_layout.addWidget(self.exit_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        

        # 在这里绑定按钮点击事件
        self.open_image_button.clicked.connect(self.select_file)
        self.save_result_button.clicked.connect(self.save_result)
        self.exit_button.clicked.connect(self.exit_app)
        self.mask_radio.toggled.connect(self.display_results)
        self.original_image_radio.toggled.connect(self.display_results)
        self.target_selection_combo.currentIndexChanged.connect(self.display_results)
        self.segmentation_result_checkbox.toggled.connect(self.display_results)
        self.detection_box_checkbox.toggled.connect(self.display_results)
        self.refresh_window1_button.clicked.connect(lambda: self.refresh_window())

    def refresh_window(self):
        '''刷新窗口内容'''
        if not hasattr(self, 'result') or not self.image_id:
            return
        self.load_and_display_image()
        self.display_results(self.result)

    def load_and_display_image(self):
        '''加载并显示图像'''
        if not self.image_id:
            self.image_label1.setText("无图像数据")
            return

        image_url = f"{Server_URL}/image/{self.image_id}"
        try:
            response = requests.post(image_url)
            if response.status_code == 200:
                image_data = response.content
                q_image = QImage()
                q_image.loadFromData(image_data)

                pixmap = QPixmap.fromImage(q_image).scaled(
                    self.image_label1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label1.setPixmap(pixmap)
            else:
                self.image_label1.setText("图像加载失败")
        except Exception as e:
            self.image_label1.setText(f"加载错误: {str(e)}")

    def display_results(self, response_json=None):
        '''显示识别结果'''
        print("Received response type:", type(response_json))
        print("Response content:", response_json)

        if not isinstance(response_json, dict):
            return
        
        
        if response_json is not None:
            # 保存响应数据和图像 ID
            self.result = response_json
            self.image_id = self.result.get('image_id')
            self.img_format = self.result.get('format', '.jpg')

        # 获取诊断总结
        diagnosis_summary = self.result.get('diagnosis_summary', [])
        if diagnosis_summary:
            html_content = ""
            for line in diagnosis_summary:
                if "🔴" in line:
                    html_content += f"<font color='red'>{line}</font><br>"
                elif "⚠️" in line:
                    html_content += f"<font color='orange'>{line}</font><br>"
                elif "🟡" in line:
                    html_content += f"<font color='gold'>{line}</font><br>"
                elif "🟢" in line:
                    html_content += f"<font color='green'>{line}</font><br>"
                else:
                    html_content += f"{line}<br>"
            self.recognition_result_text.setText(html_content)


        # 异步加载主图
        self.load_and_display_image()

        predictions = self.result.get('predictions', [])

        # 当新图像没有检测结果时，直接将窗口2设置为默认提示信息
        if not predictions:
            self.image_label2.setText("未检测到目标")
            self.reset_ui_elements()
            return
    
        # 下拉框绑定目标列表
        self.target_selection_combo.clear()
        for pred in predictions:
            self.target_selection_combo.addItem(f"目标 {pred['id']} ({pred['label']})", userData=pred)

        # 默认选择第一个目标
        selected_pred = self.target_selection_combo.currentData() if predictions else None

        # 获取窗口状态
        show_segmentation = self.segmentation_result_checkbox.isChecked()
        show_detection = self.detection_box_checkbox.isChecked()
        show_mask = self.mask_radio.isChecked()

        # 先判断是否显示检测框图像
        if show_detection:
            qimg = self.get_label_image()
            if qimg is not None:
                pixmap = QPixmap.fromImage(qimg).scaled(
                    self.image_label1.size(), Qt.KeepAspectRatio
                )

                # 如果叠加 mask
                if show_segmentation and selected_pred:
                    pixmap = self.draw_image_with_mask(pixmap, selected_pred, True)

                self.image_label1.setPixmap(pixmap)
                return

        # 否则才进入常规绘制流程（原始图像 + 可选 mask）
        qimg = self.get_original_image()
        if qimg is None:
            self.image_label1.setText("无图像数据")
            return

        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label1.size(), Qt.KeepAspectRatio)

        # 绘制 mask
        if show_segmentation and selected_pred:
            pixmap = self.draw_image_with_mask(pixmap, selected_pred, True)

        self.image_label1.setPixmap(pixmap)

        # 更新目标信息
        if selected_pred:
            bbox = selected_pred.get('bbox')
            confidence = selected_pred.get('confidence', 0)
            label = selected_pred.get('label', '未知')

            # 更新目标位置标签
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                self.xmin_label.setText(f"xmin: {x1}")
                self.ymin_label.setText(f"ymin: {y1}")
                self.xmax_label.setText(f"xmax: {x2}")
                self.ymax_label.setText(f"ymax: {y2}")
            else:
                self.xmin_label.setText("xmin: N/A")
                self.ymin_label.setText("ymin: N/A")
                self.xmax_label.setText("xmax: N/A")
                self.ymax_label.setText("ymax: N/A")

            # 更新识别结果
            self.result_text.setText(f"{label} (置信度: {confidence:.2f})")

            # 切换 window2 内容
            self.update_window2_display(selected_pred, show_mask)

        else:
            self.xmin_label.setText("xmin: N/A")
            self.ymin_label.setText("ymin: N/A")
            self.xmax_label.setText("xmax: N/A")
            self.ymax_label.setText("ymax: N/A")
            self.result_text.setText("暂无目标")
            self.image_label2.setText("未检测到目标")

        # 更新顶部信息
        self.total_targets_label.setText(f"总目标数: {len(predictions)}")
        self.time_label.setText(f"用时: {self.result.get('inference_time', 0):.2f}s")

        # 表格填充
        self.table.setRowCount(len(predictions))
        for row, pred in enumerate(predictions):
            self.table.setItem(row, 0, QTableWidgetItem(str(pred['id'])))
            self.table.setItem(row, 1, QTableWidgetItem(pred['label']))
            self.table.setItem(row, 2, QTableWidgetItem(pred['original_label']))
            self.table.setItem(row, 3, QTableWidgetItem(f"{pred['confidence']:.2f}"))
            bbox = pred['bbox']
            self.table.setItem(row, 4, QTableWidgetItem(f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"))

        # 表格设置交替行颜色
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            alternate-background-color: #f2f2f2;
            selection-background-color: #cce5ff;
        """)

        # 绑定下拉框事件（防止重复绑定）
        self.target_selection_combo.currentIndexChanged.connect(self.on_target_changed)


    def on_target_changed(self):
        """ 下拉框切换目标时刷新显示 """
        selected_pred = self.target_selection_combo.currentData()
        if not selected_pred:
            return

        # 更新目标位置
        bbox = selected_pred.get('bbox')
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            self.xmin_label.setText(f"xmin: {x1}")
            self.ymin_label.setText(f"ymin: {y1}")
            self.xmax_label.setText(f"xmax: {x2}")
            self.ymax_label.setText(f"ymax: {y2}")
        else:
            self.xmin_label.setText("xmin: N/A")
            self.ymin_label.setText("ymin: N/A")
            self.xmax_label.setText("xmax: N/A")
            self.ymax_label.setText("ymax: N/A")

        # 切换 mask / 原图
        show_mask = self.mask_radio.isChecked()
        self.update_window2_display(selected_pred, show_mask)
        

    def update_window2_display(self, selected_pred, show_mask):
        if not selected_pred:
            self.image_label2.setText("未检测到目标")
            return

        if show_mask:
            mask_id = selected_pred.get('mask_id')
            if mask_id:
                mask_url = f"{Server_URL}/image/{mask_id}"
                try:
                    response = requests.post(mask_url)
                    if response.status_code == 200:
                        mask_data = response.content
                        mask_qimg = QImage()
                        mask_qimg.loadFromData(mask_data, "PNG")
                        mask_pixmap = QPixmap.fromImage(mask_qimg).scaled(
                            self.image_label2.size(), Qt.KeepAspectRatio
                        )
                        self.image_label2.setPixmap(mask_pixmap)
                    else:
                        self.image_label2.setText("Mask 加载失败")
                except Exception as e:
                    self.image_label2.setText(f"Mask叠加失败: {str(e)}")
            else:
                self.image_label2.setText("无可用Mask")
        else:
            qimg_orig = self.get_original_image()
            if qimg_orig:
                pixmap_orig = QPixmap.fromImage(qimg_orig).scaled(
                    self.image_label2.size(), Qt.KeepAspectRatio
                )
                self.image_label2.setPixmap(pixmap_orig)
            else:
                self.image_label2.setText("图像加载失败")


    def get_label_image(self):
        """
        获取带检测框与标签的图像
        :return: QImage 对象或 None
        """
        labeled_image_id = self.result.get('labeled_image_id')
        if not labeled_image_id:
            return None

        labeled_url = f"{Server_URL}/labeled_image/{labeled_image_id}"
        try:
            response = requests.post(labeled_url)
            if response.status_code == 200:
                qimg = QImage()
                qimg.loadFromData(response.content)
                return qimg
        except Exception as e:
            print(f"❌ 获取 label_image 失败: {e}")
            return None

    def get_original_image(self):
        """
        获取原始图像（无检测框、mask）
        :return: QImage 对象或 None
        """
        if not self.image_id:
            return None

        image_url = f"{Server_URL}/image/{self.image_id}"
        try:
            response = requests.post(image_url)
            if response.status_code == 200:
                qimg = QImage()
                qimg.loadFromData(response.content)
                return qimg
        except Exception as e:
            print(f"❌ 获取原始图像失败: {e}")
            return None


    def draw_image_with_mask(self, pixmap, selected_pred, show_segmentation=False):
        """
        在 pixmap 上绘制 mask（如果需要）
        :param pixmap: 原始 QPixmap 图像
        :param selected_pred: 当前选择的目标预测数据
        :param show_segmentation: 是否叠加 mask
        return: 新的 QPixmap（可能带有 mask）
        """
        if not pixmap or not selected_pred:
            return pixmap

        # 创建 QImage 副本用于绘制
        image = pixmap.toImage().copy()
        image = image.convertToFormat(QImage.Format_RGBA8888)
        painter = QPainter(image)

        # 绘制 mask
        if show_segmentation:
            mask_id = selected_pred.get('mask_id')
            if mask_id:
                mask_url = f"{Server_URL}/image/{mask_id}"
                try:
                    response = requests.post(mask_url)
                    if response.status_code == 200:
                        mask_data = response.content
                        mask_qimg = QImage()
                        mask_qimg.loadFromData(mask_data, "PNG")
                        mask_pixmap = QPixmap.fromImage(mask_qimg).scaled(
                            pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                        painter.setOpacity(0.5)
                        painter.drawPixmap(0, 0, mask_pixmap)
                except Exception as e:
                    print(f"Mask叠加失败: {str(e)}")

        painter.end()
        return QPixmap.fromImage(image)

    def save_result(self):
        if self.image_label1.pixmap() is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先上传并处理图像后再保存结果")
            return

        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(self, "选择保存路径", options=options)
        if not folder:
            return

        # 保存图像
        window1_pixmap = self.image_label1.pixmap()
        window2_pixmap = self.image_label2.pixmap()

        if window1_pixmap:
            window1_pixmap.save(os.path.join(folder, "result_image_window1.png"), "PNG")

        if window2_pixmap:
            window2_pixmap.save(os.path.join(folder, "result_image_window2.png"), "PNG")

        # 拼接图像
        if window1_pixmap and window2_pixmap:
            try:
                img1 = window1_pixmap.toImage()
                img2 = window2_pixmap.toImage()

                combined_width = img1.width() + img2.width()
                combined_height = max(img1.height(), img2.height())

                combined_image = QImage(combined_width, combined_height, QImage.Format_RGBA8888)
                painter = QPainter(combined_image)

                painter.drawImage(0, 0, img1)
                painter.drawImage(img1.width(), 0, img2)

                painter.end()

                combined_pixmap = QPixmap.fromImage(combined_image)
                combined_pixmap.save(os.path.join(folder, "result_image_combined.png"), "PNG")
            except Exception as e:
                print(f"❌ 合并图像失败: {str(e)}")

        # 保存表格数据为 CSV
        import csv
        with open(os.path.join(folder, "results.csv"), 'w', newline='', encoding='GBK') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["序号", "中文类别","英文类别","置信度", "坐标位置"])
            for row in range(self.table.rowCount()):
                row_data = []
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)
        print("数据已保存到文件夹：", folder)
    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.dcm)",
            options=options
        )
        if file_name:
            self.segmentation_result_checkbox.setChecked(True)
            self.detection_box_checkbox.setChecked(False)
            self.mask_radio.setChecked(True) 
            self.process_file(file_name)

    #  重置 UI 元素
    def reset_ui_elements(self):
        """重置所有与预测结果相关的 UI 元素"""
        # 表格清空
        self.table.setRowCount(0)

        # 置信度与识别结果
        self.result_text.setText("暂无目标")

        # 目标位置信息
        self.xmin_label.setText("xmin: N/A")
        self.ymin_label.setText("ymin: N/A")
        self.xmax_label.setText("xmax: N/A")
        self.ymax_label.setText("ymax: N/A")

        # 下拉框清空
        self.target_selection_combo.clear()

        # 顶部信息栏
        self.total_targets_label.setText("总目标数:")
        self.time_label.setText("用时:")

        # 图像显示区
        self.image_label2.setText("未检测到目标")



    # 处理文件
    def process_file(self, file_path):
        self.image_label2.setText("正在处理...")
        self.worker = PredictionThread(file_path,parent=self)
        self.worker.finished_signal.connect(self.display_results)
        self.worker.error_signal.connect(self.show_error_message)
        self.worker.start()
    # 显示错误信息
    def show_error_message(self, message):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "错误", message)
    # 退出应用程序
    def exit_app(self):
        sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())