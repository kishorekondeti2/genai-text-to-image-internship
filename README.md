##### Text-to-Image Generator Internship Project
Author: Kishor Kondeti
Program: Learn to Build Real Time Text To Image Generator - Gen AI
Internship Provider: Nullclass (IEEE-Standard, MSME-Certified)

### Project Overview
This project implements a comprehensive text-to-image generation pipeline, fulfilling all required internship tasks. Features include:

Public dataset analysis

Text preprocessing and transformer-based embedding creation

Conditional GAN (CGAN) for basic shapes from text labels

Integration of self-attention in GANs

Fine-tuning Stable Diffusion on a custom domain

End-to-end workflow and visual outputs

All code builds on my original training project—new tasks appear as extra features/sections.

### Dataset Access
The full dataset is hosted on Google Drive due to size limitations.
Download/access it here: https://drive.google.com/file/d/1dEDdTAEersFgdl_dNks2PdVeYm8EzPrE/view?usp=drivesdk

Instructions for loading/preprocessing the dataset are included in the notebook. For questions, contact me at [kondetikishor@gmail.com].

🚀 How to Run
Recommended: Open Demo in Google Colab

Run each cell from top to bottom or use "Runtime > Run all".

Follow inline comments and Markdown for explanations.

All visuals, tables, and outputs will appear in the notebook.

Requirements:

Python 3.8+ (for local .py files)

Libraries: torch, torchvision, matplotlib, pandas, pillow, seaborn, wordcloud, transformers, diffusers

### Tasks Implemented
Task	   Description
Task 1	Comprehensive text-to-image pipeline with GAN-based image generation, text preprocessing, embedding creation
Task 2	Integration of self-attention/cross-attention in GANs for enhanced image fidelity
Task 3	Fine-tuning of a pretrained text-to-image model (Stable Diffusion) on a custom dataset
Task 4	Statistical and visual analysis of a public dataset (COCO, Oxford-102 Flowers, or custom)
Task 5	Text preprocessing and embedding creation using Hugging Face Transformers
Task 6	CGAN model generating visuals (e.g., shapes) conditionally from text labels
📊 Visual Outputs
Dataset preview: sample images and captions

Class balance and caption length statistics

Word clouds and color palette visualizations

PCA plots and similarity analysis of text embeddings

GAN/CGAN sample generations and training curves

Model comparison charts (where applicable)

📋 Methodology
Dataset Preparation: Public/CUSTOM datasets loaded, statistics computed, and visualized.

Text Preprocessing: Clean/tokenize texts, generate transformer embeddings.

Modeling: GAN/CGAN architectures; label conditioning, attention integration.

Fine-Tuning: Adapting Stable Diffusion using LoRA/PEFT for domain-specific image generation.

Evaluation: Qualitative and quantitative comparison, visual outputs.

📝 Usage Notes
If running locally, download the dataset from the Google Drive link above.

For Colab, upload the downloaded dataset to your Colab environment, following notebook instructions.

Plots, metrics, and sample images will be auto-generated and saved.

📖 Results & Insights
CGAN reliably generates basic shapes as per labels.

Attention modules help GANs focus on relevant features for better image synthesis.

Fine-tuning pretrained models on custom data leads to improved domain-specific outputs.

Embedding quality and visualizations reveal strong semantic representation.

📍 Submission Info
Project, notebook, results, and README hosted at:
https://github.com/kishorekondeti2/genai-text-to-image-internship

All tasks are implemented as new features on top of training project.

Dataset available via Google Drive.

Notebook is public and reproducible as required by Nullclass Internship guidelines.

🤝 Contact
For issues, questions, or collaboration:

Email: [kondetikishor@gmail.com]

LinkedIn: [https://www.linkedin.com/in/kondeti-kishore-594456378]

📚 References
Nullclass Internship Portal & Guidelines

Hugging Face Model Hub

PyTorch, Transformers, Diffusers, Matplotlib docs

COCO & Oxford-102 Flowers datasets

Thank you for reviewing my submission!

