import threading
import time
import uuid
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pydicom as dicom
from PIL import Image
import io
import numpy as np
import os

import torch
from ultralytics import YOLO

# 初始化 Flask 应用
app = Flask(__name__)
# 设置 CORS 允许的跨域请求
# CORS(app, resources={r"/*": {"origins": "http://your-hospital-domain.com"}})
CORS(app)
# 设置 JSONIFY 最大响应体大小
app.config['JSONIFY_MAX_SIZE'] = 10 * 1024 * 1024  # 设置为 10MB

# 添加标签映射表
label_mapping = {
    'Glioma': '胶质瘤',
    'Meningioma': '脑膜瘤',
    'Pituitary tumor': '垂体瘤'
}

# 肿瘤类型特异性阈值定义（单位：像素面积）
TUMOR_TYPE_THRESHOLDS = {
    'Glioma': {  # 胶质瘤
        'high_risk': 5000,   # 高风险面积阈值
        'medium_risk': 2000  # 中等风险面积阈值
    },
    'Meningioma': {  # 脑膜瘤
        'high_risk': 8000,
        'medium_risk': 4000
    },
    'Pituitary tumor': {  # 垂体瘤
        'high_risk': 3000,
        'medium_risk': 1500
    }
}
# 全局缓存图像（示例）
image_cache = {} # 缓存原始图像
labeled_image_cache = {}  # 缓存带检测框和标签的图像
mask_cache = {}  # mask 缓存
prediction_cache = {}  # 缓存预测数据

# 诊断总结
diagnosis_summary = []

# 加载 YOLO 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("./model.pt")  # 加载模型

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    file = request.files['file']
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()

    
    if file_ext == '.dcm':
        # 读取 DICOM 文件
        dicom_data = dicom.dcmread(file.stream,force=True)
        info20 = dicom_data.RescaleIntercept# RescaleIntercept翻译为“缩放截取”，即CT的窗位窗宽
        info21 = int(dicom_data.RescaleSlope)# RescaleSlope翻译为“缩放斜率”，即CT的窗宽
        info22 = dicom_data.pixel_array #  获取像素值
        # 得出密度值
        CT = info20 + info21 * info22  # 将像素值转换为 Hounsfield 单位

        info18 = dicom_data.WindowCenter
        info19 = dicom_data.WindowWidth
        # 计算窗位窗宽
        CT_min = info18 - info19/2
        CT_max = info18 + info19/2
        CT = np.clip(CT, CT_min, CT_max)  # 限制 CT 值在窗位窗宽范围内
        # 归一化处理
        CT_image = (CT - CT_min) / (CT_max - CT_min + 1e-5) * 255 # 防止除零错误
        # 转换为 PIL 图像
        img = Image.fromarray(CT_image.astype(np.uint8))# 转换为灰度图像
        # 缩放至最大 640x640，保持宽高比
        max_size = (640, 640)
        img.thumbnail(max_size, Image.LANCZOS)  # thumbnail 自动保持比例
        
    elif file_ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(file.stream)
        if file_ext in ['.jpg', '.jpeg']:
            file_ext = '.jpg'
        else:
            file_ext = '.png'
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG' if file_ext == '.jpg' else 'PNG')
    img_bytes = img_byte_arr.getvalue()

    # 得到阈值
    conf_threshold = float(request.form.get('conf_threshold', 0.5))
    iou_threshold = float(request.form.get('iou_threshold', 0.7))
    
    
    # 模型预测
    results = model.predict(img, 
                            conf=conf_threshold, 
                            iou=iou_threshold,
                            device=device,
                            imgsz=640)

    if not results:
        return jsonify({'error': 'No detection results'}), 400

    result = results[0]
    boxes = result.boxes
    predictions = []
    total_detections = 0

    mask_ids = []
    # 处理掩码数据
    if hasattr(result, 'masks') and result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        for i, (mask_data, box) in enumerate(zip(masks_data, boxes)):
            mask_single = (mask_data * 255).astype(np.uint8)
            orig_h, orig_w = result.orig_shape
            mask_resized = cv2.resize(mask_single, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            try:
                _, mask_encoded = cv2.imencode('.png', mask_resized)
                mask_bytes = mask_encoded.tobytes()
                mask_id = str(uuid.uuid4())
                mask_cache[mask_id] = mask_bytes
                mask_ids.append(mask_id)
            except Exception as e:
                print(f"Mask 编码失败: {e}")
                mask_ids.append(None)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        label = result.names[cls]

        chinese_label = label_mapping.get(label, label)

        width = x2 - x1
        height = y2 - y1
        total_detections += 1

        predictions.append({
            'id': idx + 1,
            'bbox': [x1, y1, x2, y2],
            'original_label': label,
            'label': chinese_label,
            'confidence': conf,
            'box_width': width,
            'box_height': height,
            'mask_id': mask_ids[idx] if idx < len(mask_ids) else None,
        })
    # 诊断总结
    diagnosis_summary = generate_diagnosis_summary(predictions)

    # 计算推理时间
    inference_time = round(time.time() - start_time, 2)

    # 缓存原始图像
    image_id = str(uuid.uuid4())
    image_cache[image_id] = img_bytes  # 原始图像 bytes

    # 使用 YOLO 的 plot 方法生成带检测框和标签的图像
    labeled_img = result.plot()  # 返回的是 numpy array (BGR 格式)
    labeled_img_pil = Image.fromarray(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))  # 转为 RGB

    # 保存为 bytes 数据
    labeled_byte_arr = io.BytesIO()
    labeled_img_pil.save(labeled_byte_arr, format=img.format if img.format else 'PNG')
    labeled_bytes = labeled_byte_arr.getvalue()

    # 缓存带标注的图像
    labeled_image_id = str(uuid.uuid4())
    labeled_image_cache[labeled_image_id] = labeled_bytes

    result_data = {
        'image_id': image_id, # 返回原始图像的 ID
        'labeled_image_id': labeled_image_id, # 返回标注图像的 ID
        'format': file_ext, # 返回原始图像的格式
        'predictions': predictions, # 返回预测结果
        'total_detections': total_detections, # 返回检测到的目标数量
        'inference_time': inference_time, # 返回推理时间
        'diagnosis_summary': diagnosis_summary # 返回诊断总结
    }

    prediction_cache[image_id] = result_data  # 用于 /labeled_image 接口获取预测信息
    return jsonify(result_data)

def generate_diagnosis_summary(predictions):
    diagnosis_summary = []
    
    if not predictions:
        diagnosis_summary.append("🟢 未检测到肿瘤迹象。")
        return diagnosis_summary
        
    # 按类型分类统计
    tumor_stats = {
        'Glioma': {'count': 0, 'max_area': 0, 'total_area': 0},
        'Meningioma': {'count': 0, 'max_area': 0, 'total_area': 0},
        'Pituitary tumor': {'count': 0, 'max_area': 0, 'total_area': 0}
    }
    
    for pred in predictions:
        tumor_type = pred['original_label']
        area = pred['box_width'] * pred['box_height']  # 计算实际面积
        
        if tumor_type in tumor_stats:
            tumor_stats[tumor_type]['count'] += 1
            tumor_stats[tumor_type]['total_area'] += area
            if area > tumor_stats[tumor_type]['max_area']:
                tumor_stats[tumor_type]['max_area'] = area
    
    # 为每种类型的肿瘤生成建议
    for tumor_type, stats in tumor_stats.items():
        if stats['count'] == 0:
            continue
            
        thresholds = TUMOR_TYPE_THRESHOLDS[tumor_type]
        max_area = stats['max_area']
        
        if max_area > thresholds['high_risk']:
            diagnosis_summary.append(f"🔴【高风险】检测到{label_mapping[tumor_type]}且最大病灶面积超过{thresholds['high_risk']}像素，建议立即进行临床评估。")
        elif max_area > thresholds['medium_risk']:
            diagnosis_summary.append(f"⚠️【中风险】检测到{label_mapping[tumor_type]}且病灶面积超过{thresholds['medium_risk']}像素，建议进一步检查。")
        else:
            diagnosis_summary.append(f"🟡【低风险】检测到较小的{label_mapping[tumor_type]}病灶，建议定期随访观察。")
    
    # 添加总体建议
    if len(predictions) > 3:
        diagnosis_summary.append("⚠️ 检测到多个病灶，可能存在广泛性病变，建议结合临床分析。")
    elif len(predictions) > 1:
        diagnosis_summary.append("🟡 检测到多发病灶，建议密切监测变化情况。")
    
    return diagnosis_summary



@app.route('/image/<image_id>', methods=['POST'])
def get_image(image_id):
    img_bytes = image_cache.get(image_id)
    if img_bytes:
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')

    mask_bytes = mask_cache.get(image_id)
    if mask_bytes:
        return send_file(io.BytesIO(mask_bytes), mimetype='image/png')

    return jsonify({'error': 'Image or Mask not found'}), 404

@app.route('/labeled_image/<image_id>', methods=['POST'])
def get_labeled_image(image_id):
    labeled_bytes = labeled_image_cache.get(image_id)
    if not labeled_bytes:
        return jsonify({'error': 'Labeled image not found'}), 404
    return send_file(io.BytesIO(labeled_bytes), mimetype='image/png')

def clear_cache():
    while True:
        time.sleep(60*60)  # 每小时清理一次
        image_cache.clear()
        mask_cache.clear()
        prediction_cache.clear()
        labeled_image_cache.clear()
        print("所有缓存已清理")

# 启动后台线程
threading.Thread(target=clear_cache, daemon=True).start()


if __name__ == '__main__':
    # 启动 Flask 应用
    #debug=True  # 开启调试模式,实际部署时应设置为 False
    #host='0.0.0.0' # 要前端访问的地址——监听所有 IPv4 接口 实际应用域名访问
    #port=5000 # 端口号
    app.run(host='0.0.0.0', port=5000, debug=True)