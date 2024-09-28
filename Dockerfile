# Menggunakan image dasar Python
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
# Set working directory
WORKDIR /app

# Salin requirements.txt ke container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file project ke dalam container
COPY . .

# Expose port 8080 yang digunakan oleh Cloud Run
EXPOSE 8080

# Set variabel environment untuk Flask
ENV FLASK_APP=yolo_api.py
ENV FLASK_ENV=production

# Jalankan Flask server
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
