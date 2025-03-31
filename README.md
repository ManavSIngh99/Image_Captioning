# Image Caption Generator Using Deep Learning

## Overview
This project implements an **Image Caption Generator** using deep learning techniques. It combines **Convolutional Neural Networks (CNNs)** for image feature extraction and **Recurrent Neural Networks (RNNs) with LSTM** for caption generation.

## Features
- Did NLP tasks on caption input.
- Uses a **pre-trained CNN model** (DenseNet201) to extract image features.
- Employs an **LSTM-based RNN model** for caption generation.
- Trained on the **Flickr8k dataset**.


## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- NLTK


### Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation
Download and extract the **Flickr8k** dataset and place it in the `data/` directory. Ensure that images and captions are properly formatted.

## Training the Model
1. **Extract Image Features:**
   ```bash
   python extract_features.py
   ```
2. **Preprocess Captions:**
   ```bash
   python preprocess_captions.py
   ```
3. **Train the Model:**
   ```bash
   python train.py
   ```

## Testing & Inference
To generate captions for an image, run:
```bash
python generate_caption.py --image_path sample.jpg
```


## Model Architecture
- **Feature Extraction:** Pre-trained CNN (DenseNet201)
- **Sequence Processing:** LSTM-based RNN
- **Embedding Layer:** Word embeddings for captions
- **Decoder:** Merges image features and text embeddings to generate captions

## Results
The model generates human-like captions for images with reasonable accuracy. Example:
**Input Image:** 🖼️ (dog playing in the park)
**Generated Caption:** *"A dog is playing with a ball in the park."*

## Future Improvements
- Implement **Attention Mechanism** for better context understanding.
- Fine-tune using **transformer-based models**.
- Extend dataset for more diverse captions.



---
**Author:** Manav Singh Jadaun
**GitHub:** [ManavSIngh99](https://github.com/ManavSIngh99)


