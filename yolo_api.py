from flask import Flask, request, jsonify
import os
from ultralytics import YOLO, RTDETR
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import uuid
import json
from werkzeug.utils import secure_filename 


app = Flask(__name__)

# Inisialisasi Firebase Admin SDK menggunakan service account JSON
cred = credentials.Certificate("tomato-diseases12-bac1c845481b.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'tomato-diseases12.appspot.com'
})

# Koneksi ke Firestore dan Storage
db = firestore.client()
bucket = storage.bucket()

# Buat folder 'uploads' jika belum ada
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# model = YOLO('models/yolov8_model.pt')

model = RTDETR('models/best.pt')

# Label penyakit tomat yang dideteksi
disease_labels = ['Healthy', 'Rotten-Tomato', 'bacterial-Spot', 'blossomendrotrotation', 'cracking', 'spliting', 'sunscaled']

# Warna bounding box untuk setiap penyakit
class_colors = {
    'bacterial-Spot': (14, 122, 254),
    'Healthy': (199, 252, 0),
    'Rotten-Tomato': (0, 183, 235),
    'blossomendrotrotation': (134, 34, 255),
    'cracking': (254, 0, 86),
    'spliting': (0, 255, 206),
    'sunscaled': (255, 128, 0)
}

CONFIDENCE_THRESHOLD = 0.5

# Endpoint untuk deteksi penyakit tomat
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
                class_index = int(box.cls)
                disease_name = disease_labels[class_index] if class_index < len(disease_labels) else None

                if disease_name in class_colors:
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    color = class_colors[disease_name]

                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                    label_text = f'{disease_name} {confidence:.2f}'
                    text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
                    text_y_min = y_min - (text_bbox[3] - text_bbox[1])
                    draw.rectangle([x_min, text_y_min, x_max, y_min], fill=color)
                    draw.text((x_min, text_y_min), label_text, font=font, fill=(255, 255, 255))

                    detection_list.append({
                        "disease": disease_name,
                        "confidence": confidence,
                        "box": [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    })
                    detected_diseases.add(disease_name)
                    num_boxes_detected += 1

    # Proses hasil deteksi
    if num_boxes_detected > 0 and 'Healthy' in detected_diseases and len(detected_diseases) == 1:
        kondisi_tomat = "Healthy"
        keterangan = "Tomat Anda sedang tidak dalam Kondisi terjangkit penyakit"
    elif num_boxes_detected > 0:
        kondisi_tomat = "Not Healthy"
        penyakit_terdeteksi = ', '.join([disease for disease in detected_diseases if disease != 'Healthy'])
        keterangan = f"Tomat Anda saat ini terdeteksi terkena penyakit {penyakit_terdeteksi}"
    else:
        kondisi_tomat = "Bukan Gambar Tomat"
        keterangan = "Aplikasi ini hanya digunakan untuk mengecek kondisi tomat"
        os.remove(image_path)

        return jsonify({
            "Kondisi Tomat": kondisi_tomat,
            "Keterangan": keterangan,
            "num_boxes_detected": num_boxes_detected
        })

    # Simpan gambar hasil deteksi ke Firebase Storage
    output_image_path = os.path.join("uploads", f"detected_{image.filename}")
    img.save(output_image_path)

    storage_filename = f"detected_images/{uuid.uuid4()}.jpg"
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(output_image_path)
    blob.make_public()
    image_url = blob.public_url

    detection_data = {
        "timestamp": datetime.now(),
        "Kondisi Tomat": kondisi_tomat,
        "Keterangan": keterangan,
        "num_boxes_detected": num_boxes_detected,
        "detections": detection_list,
        "image_url": image_url
    }

    db.collection('tomato_detections').add(detection_data)

    os.remove(image_path)
    os.remove(output_image_path)

    return jsonify({
        "Kondisi Tomat": kondisi_tomat,
        "Keterangan": keterangan,
        "num_boxes_detected": num_boxes_detected,
        "detections": detection_list,
        "image_url": image_url
    })

# Endpoint untuk membuat artikel baru
@app.route('/articles', methods=['POST'])
def create_article():
    if 'image' not in request.files:
        return jsonify({"error": "Gambar harus diupload"}), 400

    image = request.files['image']
    data = request.form.get('data')

    if not data:
        return jsonify({"error": "Data artikel tidak ditemukan"}), 400

    try:
        # Parse JSON dari data
        article_data = json.loads(data)
        title = article_data['title']
        content = article_data['content']
        author = article_data['author']
        publish_date = article_data['publishDate']
        tags = article_data['tags']

        # Simpan gambar ke storage Firebase
        image_filename = secure_filename(image.filename)
        image_path = os.path.join('uploads', image_filename)
        image.save(image_path)

        blob = bucket.blob(f"articles/{uuid.uuid4()}_{image_filename}")
        blob.upload_from_filename(image_path)
        blob.make_public()
        image_url = blob.public_url

        # Simpan data artikel ke Firestore
        article_ref = db.collection('articles').add({
            'title': title,
            'content': content,
            'author': author,
            'publishDate': publish_date,
            'tags': tags,
            'imageUrl': image_url
        })

        # Hapus file gambar dari folder uploads
        os.remove(image_path)

        return jsonify({
            "message": "Artikel berhasil diposting",
            "id": article_ref[1].id,
            "article": {
                "title": title,
                "content": content,
                "author": author,
                "publishDate": publish_date,
                "tags": tags,
                "imageUrl": image_url
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk mengambil semua artikel
@app.route('/articles', methods=['GET'])
def get_articles():
    try:
        articles_ref = db.collection('articles')
        docs = articles_ref.stream()
        articles = []
        
        for doc in docs:
            article = doc.to_dict()
            article['id'] = doc.id  # Tambahkan ID dokumen ke data artikel
            articles.append(article)

        return jsonify(articles), 200
    except Exception as e:
        return jsonify({"error": f"Error retrieving articles: {str(e)}"}), 500


# Endpoint untuk mengambil artikel berdasarkan ID
@app.route('/articles/<article_id>', methods=['GET'])
def get_article_by_id(article_id):
    try:
        # Dapatkan artikel berdasarkan ID
        doc_ref = db.collection('articles').document(article_id)
        doc = doc_ref.get()
        if doc.exists:
            article = doc.to_dict()
            article['id'] = article_id  # Tambahkan ID ke data artikel
            return jsonify(article), 200
        else:
            return jsonify({"error": "Artikel tidak ditemukan"}), 404
    except Exception as e:
        return jsonify({"error": f"Error retrieving article: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
