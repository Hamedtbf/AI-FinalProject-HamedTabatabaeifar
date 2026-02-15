FROM tensorflow/tensorflow:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --ignore-installed

COPY . .

RUN mkdir -p static/results

EXPOSE 5000

CMD ["python", "app.py"]