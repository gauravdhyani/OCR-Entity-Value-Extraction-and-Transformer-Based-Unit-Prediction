# OCR-Based Entity Value Extraction and Unit Prediction Using Fine-Tuned T5 Transformers

## Introduction
This project focuses on extracting entity values from images using **Optical Character Recognition (OCR)** techniques and predicting the associated quantities and units.  
Given the diverse text alignments and formats in the dataset images, we required a robust pipeline that can handle noisy, irregular layouts while producing accurate structured outputs.

---

## Approach Overview
The system is divided into two major components:

1. **Text Extraction** → Using OCR to extract text from product images.  
2. **Entity Value Prediction** → Using machine learning models to predict entity values and associated units from extracted text.

---

## Dataset Description
- **Size:** ~250,000 samples (training + testing).  
- **Contents:** Each sample includes:
  - Image links
  - Group IDs
  - Entity names
  - Entity values  

### **Challenges**
- Images vary in format, alignment, and clarity.
- Noise and irrelevant characters in OCR output.

### **Preprocessing**
- Resizing and normalization of images.
- Regex-based cleaning to remove garbage text.

---

## Challenges and Limitations 

1. **Text Complexity:** OCR struggled with diverse alignments, multi-line text, varying fonts, and noisy backgrounds.  
2. **Scale of Processing:** Running OCR on 250,000 images was resource-intensive, requiring parallelization, GPU acceleration, and checkpointing.  
3. **Training on Large OCR Outputs:** Handling large OCR datasets for T5-small demanded careful memory management, optimized batch sizes, and GPU utilization.  
4. **Resource Constraints:** Balancing accuracy and efficiency led to choosing EasyOCR and T5-small over heavier models like LLMs.  

---

## Text Extraction with OCR
We experimented with multiple OCR methods before finalizing the best choice:

- **PyTesseract** → Popular, but failed on complex layouts and small fonts.  
- **VGG16 + PyTesseract** → Region detection + OCR, but results were only marginally better.  
- **Keras-OCR** → High accuracy, but computationally too slow for large-scale usage.  
- **EasyOCR (Final Choice)** → Balanced accuracy and speed, robust for multi-line and mixed-alignment text.  

---

## Entity Value Prediction
After extracting text, predicting correct values with units was the next challenge:

- **Large Language Models (LLMs)** → Too heavy and computationally intensive for this task.  
- **BERT** → Good for numeric prediction, but weak in unit classification.  
- **T5 Transformer (Final Choice)** →  
  - Sequence generation model (**T5-small**, 70M parameters).  
  - Fine-tuned to predict both numbers and units.  
  - Lightweight and accurate after domain-specific fine-tuning.  

---

## Implementation
### 1. OCR Setup
- Text extraction with **EasyOCR**.  
- Added logging mechanism to monitor per-image processing and debugging.  

### 2. T5 Transformer Fine-Tuning
- **Model:** T5-small.  
- **Training:** Fine-tuned on labeled dataset of entity values.  
- **Optimization:** Focused on sequence generation accuracy (numbers + units).  

---

## Experiments and Results
### OCR Experiments
- **PyTesseract** → Poor accuracy, failed on small/multi-line text.  
- **VGG16 + PyTesseract** → Slight improvements, still weak.  
- **Keras-OCR** → Good accuracy, impractically slow.  
- **EasyOCR** → Best trade-off (accuracy + speed).  

### Prediction Experiments
- **LLMs** → Too computationally heavy.  
- **BERT** → Only numeric predictions.  
- **T5 Transformer** → Predicted both values + units reliably.  

### Results

- **F1 Score:** `0.3544` on validation/test set.  
- Demonstrates **feasibility of a large-scale OCR + NLP pipeline**.  
- Performance highlights:
  - Challenges of **noisy OCR outputs** and **diverse text layouts**.
  - Shows that **Transformers can generalize reasonably well** despite imperfect OCR input.

---
 
## Conclusion
By combining:
- **EasyOCR** for robust text extraction, and  
- **Fine-tuned T5-small** for accurate entity value + unit prediction,  

we built an efficient pipeline that handles the complexity of diverse retail product images while remaining computationally feasible.  

This solution is **scalable and adaptable** for large-scale entity extraction tasks in real-world datasets.

