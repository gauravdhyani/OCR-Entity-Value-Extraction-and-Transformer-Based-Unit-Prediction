# Methodology

The methodology of this project is designed to **extract entity values from images using OCR (Optical Character Recognition)** and **predict the associated quantities and units using a fine-tuned T5 transformer model**.  
The pipeline consists of **four main stages**:

1. **Data Preprocessing**  
2. **OCR-Based Feature Extraction**  
3. **Entity Value Prediction with Transformers**  
4. **Prediction Output Generation**  

---

## 1. Data Preprocessing
- **Input Data:**  
  The dataset consists of training and test CSV files containing:
  - `image_link`
  - `entity_name`
  - `entity_value`

- **Entity Value Filtering:**  
Any invalid entries are removed.

- **Standardization:**  
Images are resized to a fixed resolution (**600×600**) to reduce variation and speed up processing.

 This step ensures **clean, uniform data** for downstream OCR and prediction tasks.

---

## 2. OCR-Based Text Extraction
- **Tool Selection:**  
After evaluating multiple OCR methods:
- PyTesseract
- VGG16 + Tesseract
- Keras-OCR
- **EasyOCR (Final Choice)** → Best trade-off between speed and accuracy.

- **Image Handling:**
- Images are loaded from URLs and converted to OpenCV format.
- Downscaling step reduces image dimensions for performance.

- **Parallel Processing:**  
Used `ThreadPoolExecutor` for multi-threaded OCR processing, with progress tracked via `tqdm`.

- **Logging:**  
Logs both successful extractions and failures for debugging and performance monitoring.

- **Output:**  
For each image, recognized text is concatenated into a string and stored in a new column `output`.

Generated files:
- `Train_output.csv` → OCR text for training data.
- `Test_output.csv` → OCR text for test data.

---

## 3. Entity Value Prediction with T5 Transformer
- **Model Choice:**  
- LLMs → Too heavy and complex.
- BERT → Could predict numbers but failed on units.
- **T5-small (Final Choice)** → Lightweight, sequence-to-sequence model (70M parameters).

- **Tokenizer & Model:**  
Used HuggingFace `T5Tokenizer` and `T5ForConditionalGeneration`.

- **Input Construction:**  
Each training instance formatted as:
Input: OCR text
Output:
- **Dataset Class:**  
Custom PyTorch `Dataset` for tokenized inputs and outputs.

- **Training Strategy:**
- Fine-tuned for **3 epochs**, batch size = 8.
- Optimizer: **Adam** with weight decay + warmup steps.
- Mixed-precision training (**fp16**) for faster GPU computations.
- Checkpoints saved every 1000 steps.
- Custom callback for logging loss and learning rate.

Result: A fine-tuned T5 model capable of generating both **numerical values and associated units**.

---

## 4. Prediction Pipeline
- **Model Loading:**  
Reload fine-tuned T5 model and tokenizer from checkpoints.

- **Batch Inference:**  
Custom `EntityDataset` + `DataLoader` for efficient batch predictions.

- **Post-Processing:**
- Split predictions into `<number> <unit>` format.
- Normalize units using a mapping dictionary (e.g., `cm`, `centimeter`, `centimetre` → `cm`).
- Invalid or empty predictions recorded as blank.

- **Final Output:**  
Predictions added to test dataframe and exported as:
 ```
 OUTPUT.csv
 ```

---

## 5. Sanity Check
A validation script (`sanity.py`) checks:
- Predictions follow expected structure.
- Correctness before submission.

---

## Summary
- **OCR Extraction:** EasyOCR efficiently handles diverse text layouts.  
- **Transformer Prediction:** Fine-tuned T5-small predicts entity values with units.  
- **Output:** A clean prediction file (`OUTPUT.csv`) ready for evaluation.  