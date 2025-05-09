# Braille-Recognition

Communication is a fundamental human right, yet millions of visually impaired individuals
 face significant challenges in expressing and sharing their thoughts, especially those who
 are also unable to speak. For many, Braille serves as a crucial tool for reading and writing.
 However, Braille literacy rates remain low due to a lack of resources, and most non Braille
 readers struggle to understand it. This creates a barrier, preventing seamless communication
 between visually impaired individuals and the broader community.
 
 This project aims to bridge this gap by developing an advanced deep-learning system
 that can automatically recognize Braille from images and convert it into meaningful English
 sentences. Using Convolutional Neural Networks (CNNs), our model will identify Braille
 characters from scanned or photographed documents, allowing individuals to communicate
 more easily with those who do not understand Braille. By incorporating Natural Language
 Processing (NLP), we aim to ensure the translated sentences are grammatically correct and
 contextually meaningful.
 
 The significance of this project goes beyond technological innovation—it is about inclu
sivity. With a reliable Braille-to-English translation system, individuals who rely on Braille
 as their primary mode of communication will have greater independence in expressing their
 thoughts, sharing ideas, and participating more fully in society. Our goal is to create a
 tool that empowers those with visual and speech impairments, fostering a world where
 communication barriers no longer define a person’s ability to connect with others.


# Architecture
 The architecture of the project is using Yolo v11 to detect the braille character from the full-page braille book, 
 then using CNN transfer learning models to classify them, then apply NLP tecknique to improve the final output, which make it readable and sensible.

 # Braille_Recognition

A compact toolkit for detecting and translating Braille from images using CNN transfer learning and YOLO object detection.

## Features
- **Data Preparation**  
  – `generate_train_val_test_split.ipynb` to create train/val/test splits.  
- **Transfer Learning**  
  – `Transfer_Learning_with_earlystopping.ipynb` trains CNNs (e.g. MobileNetV2) on Braille character images.  
- **YOLO Detection**  
  – `Yolo_with_TL (2).ipynb` fine-tunes YOLOv11 Nano to locate Braille dots on full-page scans.  
- **End-to-End Pipeline**  
  – `BrailleDetection.ipynb` and `Project_part3_new.ipynb` demonstrate full inference (detection → classification → NLP correction).  
- **Pretrained Models**  
  – Check `saved_models/` for `.pt` weights ready for inference.

## Quick Start

1. **Clone & install**  
   ```bash
   git clone https://github.com/codingharry123/Braille_Recognition.git
   cd Braille_Recognition
   pip install -r requirements.txt

├── dataset/                         # Raw & annotated Braille images
├── saved_models/                    # Trained `.pt` model weights
├── src/                             # Helper scripts & modules
├── generate_train_val_test_split.ipynb
├── Transfer_Learning_with_earlystopping.ipynb
├── Yolo_with_TL (2).ipynb
├── BrailleDetection.ipynb
├── Project_part3_new.ipynb
├── References.md                    # Citations & resources
└── requirements.txt
