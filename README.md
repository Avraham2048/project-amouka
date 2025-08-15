# 📜 OCR & Diacritization Project By Tsuriel Vizel And Avraham Guez

<a name="project"></a>

Welcome to the **OCR & Diacritization** project! 🎯  
This repository contains all the data, scripts, and trained models used to build and run two separate AI models:  
1. **OCR Model** – to recognize text from images with 93 % accuracy.  
2. **Diacritization Model** – to add *nikkud* (vowel marks) to Hebrew text with 93 % accuracy.  

Below you’ll find a full breakdown of the repository structure and the purpose of each folder and file.  

---

## 📑 Table of Contents
- [Project Overview](#project)
- [📂 Project Structure](#project-structure)
  - [DiacData](#diacdata)
  - [OcrData](#ocrdata)
  - [Models](#models)
  - [UseModels](#usemodels)
  - [tools](#tools)
- [🐍 Main Python Scripts](#main-python-scripts)
- [🔄 Typical Workflow](#typical-workflow)
- [💡 Usage](#usage)
- [📝Concrete Example](#concrete-example)
- [📚 References](#references)

---

<a name="project-structure"></a>
## 📂 Project Structure

<a name="diacdata"></a>
### 1. **DiacData** 📚  
Contains all the **text data** used to train the *Diacritization Model*.

- Holds **two versions** of Hebrew text:  
  1. **With nikkud** ✅ (vowel marks).  
  2. **Without nikkud** ❌ (plain consonantal Hebrew).  
- The model learns from these **paired examples** how to insert the correct diacritical marks into plain text.

🛠 **How we built the dataset:**  
We prepared the diacritization dataset from scratch:  
1. **Download source texts** in `.txt` format from **[sefaria.org](https://www.sefaria.org)** — a large online library of Hebrew texts.  
2. Keep the **original version with nikkud** as the *target* text for the model.  
3. Run our custom Python script **`EraseDiac.py`** to remove all diacritics (nikkud) from the text, producing the *input* text.  
4. Store both the *with-nikkud* and *without-nikkud* versions in `DiacData`.

📌 **Why this matters:**  
- Hebrew without nikkud can be highly ambiguous — the same word can have many possible readings.  
- By training the model on side-by-side examples of text **with and without** diacritics, it learns to restore the correct nikkud automatically.  
- This makes the system very useful for restoring pronunciation guides in educational, liturgical, or linguistic contexts.

✨ The resulting dataset serves as the **foundation** for the `Diacmodel.py` training script.

---

<a name="ocrdata"></a>
### 2. **OcrData** 🖼️  
Stores the **training and testing data** for the OCR model.  

- Contains **images of Hebrew text** 📄 and their corresponding label files.  
- These datasets are used by `OcrModel.py` to train the OCR model to recognize **printed or handwritten Hebrew text**.  

🛠 **How we built the dataset:**  
We created our own **custom “home-made” dataset** from Hebrew books.  
1. We downloaded Hebrew texts in `.txt` format from **[sefaria.org](https://www.sefaria.org.il/texts)**.  
2. Using the Python library **[trdg](https://github.com/Belval/TextRecognitionDataGenerator)** (*Text Recognition Data Generator*), we transformed these raw text files into a large collection of synthetic training images.  
3. This process allowed us to create a dataset of **~10,000 samples** 📦.  

📌 **Command used to generate the dataset**:  
```bash
trdg -i hebrew_1.txt -fd fonts/ -c 10000 -l he -w 1 --output_dir output1/
```
Where:

- `-i hebrew_1.txt` → Input Hebrew text file.  
- `-fd fonts/` → Folder containing Hebrew fonts.  
- `-c 10000` → Number of images to generate.  
- `-l he` → Language code for Hebrew.  
- `-w 1` → Words per image (1 in this case).  
- `--output_dir output1/` → Output folder for generated images.  

✨ This gave us full control over the dataset, including the font styles, image size, and number of samples, making our OCR model more robust and accurate.

---

<a name="models"></a>
### 3. **Models** 🏆  
This is where the **trained models** are stored after training is complete.  
- Each model is saved in a format ready to be loaded for inference.  
- **Contents:**  
  - `OcrModel` (final OCR model)  
  - `DiacModel` (final diacritization model)  

---

<a name="usemodels"></a>
### 4. **UseModels** 🚀  
Contains Python scripts to **use the trained models** for real-world tasks.  
- Scripts to load and run **only the OCR model** 🖋️  
- Scripts to load and run **only the Diacritization model** ✨  
- Scripts to **chain both models** so you can:
  1. Recognize text from an image (OCR).  
  2. Automatically add vowel marks (Diacritization).  

This folder is your **go-to** when you want to *use* the models rather than train them.

---

<a name="tools"></a>
### 5. **tools** 🛠️  
A collection of **utility scripts** used throughout the project.  
- Functions for data preprocessing.  
- Helper scripts for training, evaluation, and formatting datasets.  
- Custom Python tools to simplify repetitive tasks.

---

<a name="main-python-scripts"></a>
## 🐍 Main Python Scripts

### **`OcrModel.py`** 📄  
- Script to train the **OCR model**.  
- Loads `OcrData` dataset, preprocesses it, trains the model, and saves it into `Models`.
- Hyperparameters we used:
 ```bash
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
IMAGE_HEIGHT = 64 # Resize all images to this height
IMAGE_WIDTH = 256 # Resize all images to this width
NUM_WORKERS = 4 # For DataLoader
TRAIN_VAL_SPLIT = 0.9 # 90% for training, 10% for validation
```

### **`Diacmodel.py`** ✒️  
- Script to train the **Diacritization model**.  
- Uses `DiacData` (text pairs) to learn how to insert nikkud into plain Hebrew text.  
- Saves the trained model into `Models`.
- Hyperparameters we used:
 ```bash
embed_dim = 128
hidden_dim = 256
num_layers = 2
dropout = 0.3
epochs = 20
batch_size = 16
lr = 0.0001 #We used LROnPlateau nut most of the training was with lr=0.0001
```

---

<a name="typical-workflow"></a>
## 🔄 Typical Workflow

1. **Prepare Data**  
   - Place OCR training images inside `OcrData`.  
   - Place diacritization text data inside `DiacData`.  

2. **Train Models**  
   - Run `OcrModel.py` → generates OCR model.  
   - Run `Diacmodel.py` → generates diacritization model.  

3. **Use Models**  
   - Go to `UseModels` and run the script for the task you need (OCR only, Diacritization only, or both combined).  

---

<a name="usage"></a>
## 💡 Usage
```bash
# Run OCR on an image
python UseModels/use_ocr.py  # change path to picture in the script

# Add diacritics to text
python UseModels/use_diac.py "שלום עולם"

# Full pipeline: OCR + Diacritics
python UseModels/usemodels.py 
```

---

<a name="concrete-example"></a>
## 📝Concrete Example
| Step | Input / Output | Description |
|------|---------------|-------------|
| 🖼️ Input Image | ![Example Hebrew](OcrData/test.jpg) | A Hebrew word image with no diacritics. |
| ✍️ OCR Output | חכמה | The OCR model recognizes the text without nikkud. |
| 🎯 Final Output | חָכְמָה | The diacritization model adds the correct vowel marks. |

---

📌 The beauty of this pipeline is that **each model can be used independently** or **together** depending on your needs.

---

<a name="references"></a>
## 📚 References
- [Sefaria – A Living Library of Jewish Texts](https://www.sefaria.org)  
- [TRDG – TextRecognitionDataGenerator (GitHub)](https://github.com/Belval/TextRecognitionDataGenerator)  
- [Python Official Documentation](https://docs.python.org/3/)  
- [PyTorch — Tensors and Deep Learning (Official)](https://pytorch.org)

---

**Thank you for reading us.**

***Tsuriel and Avraham.***
