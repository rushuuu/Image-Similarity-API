# Image Similarity API

A FastAPI-based web service that compares two images and returns a similarity score and category using pre-trained deep learning models (`EfficientNet-B0` and `ViT-B-16`).

---

## Features

- Upload two image files (JPG/PNG) and receive a similarity score.
- Categorizes similarity into:
  - SAME
  - ALMOST SAME
  - VERY SIMILAR
  - SLIGHTLY SIMILAR
  - NOT SIMILAR
- Uses deep learning models from PyTorch (`torchvision.models`).
- Supports GPU if available.

---

## Models Used

- EfficientNet B0
- Vision Transformer (ViT B-16)

These models are used to extract feature embeddings from input images, and cosine similarity is computed between the embeddings.

---

## 🛠️ Setup

### Clone the Repository

git clone https://github.com/your-username/image-similarity-api.git
cd image-similarity-api

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

## API Endpoints

GET /
Returns a welcome message and available endpoints.

GET /health
Returns the health status and loaded models.

POST /image_similarity/
Upload two images to compare.



