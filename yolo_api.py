from flask import Flask, request, jsonify, send_file
import os
from ultralytics import YOLO, RTDETR  
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import uuid

app = Flask(__name__)

# Inisialisasi Firebase Admin SDK menggunakan path ke service account JSON
cred = credentials.Certificate("tomato-diseases12-bac1c845481b.json")  
firebase_admin.initialize_app(cred, {
    'storageBucket': 'tomato-diseases12.appspot.com'  
})

# Buat koneksi ke Firestore dan Storage
db = firestore.client()
bucket = storage.bucket()

# Buat folder 'uploads' jika belum ada
if not os.path.exists('uploads'):
    os.makedirs('uploads')

model = YOLO('models/yolov8_model.pt')

# model = RTDETR('models/best.pt')

disease_labels = ['Healthy', 'Rotten-Tomato', 'bacterial-Spot', 'blossomendrotrotation', 'cracking', 'spliting', 'sunscaled']

class_colors = {
    'bacterial-Spot': (14, 122, 254), 
    'Healthy': (199, 252, 0),  
    'Rotten-Tomato': (0, 183, 235),  
    'blossomendrotrotation': (134, 34, 255),  
    'cracking': (254, 0, 86),  
    'spliting': (0, 255, 206),  
    'sunscaled': (255, 128, 0)  
}

CONFIDENCE_THRESHOLD = 0.3

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    # Jalankan YOLOv8 pada gambar
    results = model.predict(source=image_path)

    # Cek apakah ada deteksi yang valid
    num_boxes_detected = 0
    detection_list = []
    detected_diseases = set()
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Jika ingin menggunakan font khusus untuk label (optional)
    try:
        font = ImageFont.truetype("arial.ttf", 16)  
    except IOError:
        font = ImageFont.load_default()  

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)  

            if confidence >= CONFIDENCE_THRESHOLD:
                class_index = int(box.cls)  # Indeks kelas
                disease_name = disease_labels[class_index] if class_index < len(disease_labels) else None

                # Hanya gambarkan bounding box jika disease_name ditemukan di class_colors
                if disease_name in class_colors:
                    # Ambil koordinat bounding box
                    x_min, y_min, x_max, y_max = box.xyxy[0]

                    # Tentukan warna untuk bounding box dan teks berdasarkan kelas penyakit
                    color = class_colors[disease_name]

                    # Gambarkan bounding box di atas gambar dengan warna yang sesuai
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

                    # Tambahkan label penyakit dan confidence di atas bounding box dengan warna yang sesuai
                    label_text = f'{disease_name} {confidence:.2f}'

                    # Gunakan textbbox untuk mendapatkan ukuran teks (kiri atas, kanan bawah)
                    text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)

                    # Adjust posisi teks ke atas box
                    text_y_min = y_min - (text_bbox[3] - text_bbox[1])  

                    # Gambarkan latar belakang teks dengan warna yang sama dengan box di atas bounding box
                    draw.rectangle([x_min, text_y_min, x_max, y_min], fill=color)

                    # Gambarkan teks dengan warna putih di atas bounding box
                    draw.text((x_min, text_y_min), label_text, font=font, fill=(255, 255, 255))

                    # Simpan data deteksi
                    detection_list.append({
                        "disease": disease_name,
                        "confidence": confidence,
                        "box": [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    })
                    detected_diseases.add(disease_name)  # Simpan nama penyakit yang terdeteksi
                    num_boxes_detected += 1

    # Proses hasil deteksi
    if num_boxes_detected > 0 and 'Healthy' in detected_diseases and len(detected_diseases) == 1:
        # Jika hanya deteksi "Healthy" tanpa penyakit lain
        kondisi_tomat = "Healthy"
        keterangan = "Tomat Anda sedang tidak dalam Kondisi terjangkit penyakit"
    elif num_boxes_detected > 0:
        # Jika ada penyakit yang terdeteksi selain Healthy
        kondisi_tomat = "Not Healthy"
        penyakit_terdeteksi = ', '.join([disease for disease in detected_diseases if disease != 'Healthy'])
        keterangan = f"Tomat Anda saat ini terdeteksi terkena penyakit {penyakit_terdeteksi}"
    else:
        # Jika tidak ada deteksi apapun, tidak disimpan
        kondisi_tomat = "Bukan Gambar Tomat"
        keterangan = "Aplikasi ini hanya digunakan untuk mengecek kondisi tomat"

        # Hapus file gambar asli (bukan hasil deteksi) setelah proses selesai
        os.remove(image_path)

        return jsonify({
            "Kondisi Tomat": kondisi_tomat,
            "Keterangan": keterangan,
            "num_boxes_detected": num_boxes_detected
        })

    # Simpan gambar hasil deteksi ke Firebase Storage
    output_image_path = os.path.join("uploads", f"detected_{image.filename}")
    img.save(output_image_path)

    # Buat unique filename untuk storage
    storage_filename = f"detected_images/{uuid.uuid4()}.jpg"
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(output_image_path)
    blob.make_public()

    image_url = blob.public_url  # URL gambar yang diupload ke Firebase Storage

    # Simpan hasil ke Firestore hanya jika ada deteksi
    detection_data = {
        "timestamp": datetime.now(),
        "Kondisi Tomat": kondisi_tomat,
        "Keterangan": keterangan,
        "num_boxes_detected": num_boxes_detected,
        "detections": detection_list,
        "image_url": image_url  
    }
    
    # Menyimpan data deteksi ke koleksi Firestore
    db.collection('tomato_detections').add(detection_data)

    # Hapus file gambar asli dan hasil deteksi setelah data diupload
    os.remove(image_path)
    os.remove(output_image_path)

    return jsonify({
        "Kondisi Tomat": kondisi_tomat,
        "Keterangan": keterangan,
        "num_boxes_detected": num_boxes_detected,
        "detections": detection_list,
        "image_url": image_url  
    })

@app.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    return send_file(os.path.join('uploads', filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
