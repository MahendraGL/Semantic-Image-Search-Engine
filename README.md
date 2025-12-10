# Semantic Image Search Engine 🔍🖼️

A deep learning-based search engine that finds visually similar images using **ResNet50** for feature extraction and **FAISS** for efficient vector similarity search.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-green)

## 📌 Overview

This project implements an end-to-end semantic search pipeline. Unlike traditional keyword search, this engine "sees" the content of an image. It converts images into high-dimensional embedding vectors and uses a vector database to find the nearest neighbors (most similar images).

**Use Cases:** E-commerce product recommendation, digital asset management, stock photo search, and duplicate image detection.

## 🚀 Key Features

* **Deep Learning Backbone:** Uses **ResNet50** (pre-trained on ImageNet) to extract 2048-dimensional feature vectors.
* **Vector Database:** Implements **FAISS (Facebook AI Similarity Search)** for high-speed indexing and retrieval.
* **Similarity Metric:** Uses **Cosine Similarity** (via L2 Normalization + Inner Product) to measure visual closeness.
* **Modular Codebase:** Cleanly separated logic for feature extraction, indexing, and searching.

## 📂 Repository Structure

```text
Semantic-Image-Search-Engine/
│
├── data/
│   ├── images/          # Place your dataset images here
│   └── metadata/        # Stores the generated vector.index and metadata.pkl
│
├── src/
│   ├── feature_extractor.py   # ResNet50 logic
│   ├── index_images.py        # Script to process images & build DB
│   └── search.py              # Script to query the DB
│
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

🛠️ Installation
Clone the repository

git clone [https://github.com/MahendraGL/Semantic-Image-Search-Engine.git](https://github.com/MahendraGL/Semantic-Image-Search-Engine.git)
cd Semantic-Image-Search-Engine
Create a Virtual Environment (Optional but Recommended)

Bash

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
Install Dependencies

Bash

pip install -r requirements.txt
🏃 Usage
1. Prepare Data
Download your image dataset (e.g., from Kaggle or your personal collection) and place the images (.jpg, .png, .jpeg) inside the data/images/ folder.

2. Build the Index
Run the indexing script. This will read all images, generate embeddings using ResNet50, and save the FAISS index.

Bash

python src/index_images.py
Output: This creates vector.index and metadata.pkl in data/metadata/.

3. Search
Query the database by providing a path to any image (it can be an image from the dataset or a completely new one).

Bash

python src/search.py "path/to/test_image.jpg"
🧠 How It Works
Preprocessing: Images are resized to 256x256, center-cropped to 224x224, and normalized using standard ImageNet mean/std values.

Feature Extraction: The classification layer of ResNet50 is removed. We use the output of the final pooling layer to get a 2048 dimension vector for every image.

Indexing: These vectors are normalized (L2) and added to a IndexFlatIP FAISS index.

Retrieval: When you search, the query image undergoes the same transformation. We calculate the dot product between the query vector and all database vectors to find the closest matches.

📋 Requirements
Python 3.8+

torch

torchvision

faiss-cpu

numpy

Pillow

tqdm
