# Semantic Image Search Engine

A deep learning-based image search engine that finds visually similar images using **ResNet50** for feature extraction and **FAISS** (Facebook AI Similarity Search) for efficient vector similarity search.

## Overview

This project implements a semantic search pipeline:
1. **Feature Extraction**: Uses a pre-trained ResNet50 model (with the classification head removed) to convert images into 2048-dimensional embedding vectors.
2. **Vector Database**: Indexes these vectors using FAISS with L2 normalization (Cosine Similarity).
3. **Inference**: Allows users to query the database with a new image to find the top K nearest neighbors.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/MahendraGL/Semantic-Image-Search-Engine.git](https://github.com/MahendraGL/Semantic-Image-Search-Engine.git)
   cd Semantic-Image-Search-Engine
