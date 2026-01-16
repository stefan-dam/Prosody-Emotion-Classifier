# Prosody-Emotion-Classifier

AURA Technical Stream Prosody-Based Emotion Classifier using HuBERT for emotion classification from speech audio.

## Overview

Implements a prosody-based emotion classifier using the HuBERT (Hidden-Unit BERT) model from Hugging Face Transformers. It supports emotion classification on datasets like RAVDESS and CREMAD, mapping emotions to a common 7-label set (neutral, happy, angry, sad, disgust, fear, excited) and providing VAD (Valence-Arousal-Dominance) mappings.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- Sufficient disk space for datasets (RAVDESS ~200MB, CREMAD ~1GB+)

## Setup Instructions

### Step 1: Clone/Download the Repository

If you have the repository as a Git repository:
```bash
git clone <repository-url>
cd Prosody-Emotion-Classifier
```

Otherwise, navigate to the project directory.

### Step 2: Create a Python Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid dependency conflicts:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and TorchAudio
- Transformers (Hugging Face)
- Librosa (audio processing)
- SoundFile (audio I/O)
- Pandas, NumPy, SciPy
- Scikit-learn
- And other dependencies

**Note:** If you have a CUDA-capable GPU and want to use it for training, you may need to install PyTorch with CUDA support separately. 

### Step 4: Download Datasets

You need to download the emotion speech datasets. This project supports **RAVDESS** and **CREMAD** datasets.

#### Download RAVDESS Dataset

1. Download from the link under data/RAVDESS and extract under data directory

**Expected structure:**
```
data/
└── RAVDESS/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   ├── 03-01-01-01-01-02-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

#### Download CREMAD Dataset

1. Download from the link under data/CREMAD and extract under data directory


**Expected structure:**
```
data/
└── CREMAD/
    ├── 1001_DFA_ANG_XX.wav
    ├── 1001_DFA_DIS_XX.wav
    ├── 1001_DFA_FEA_XX.wav
    └── ...
```

**Note:** You can use one or both datasets. The project will work with either, but both provide better training data diversity.

### Step 5: Generate Metadata Files

After placing the audio files in the correct directories, generate metadata CSV files for each dataset:

**For RAVDESS:**
```bash
python src/make_metadata_ravdess.py
```

**For CREMAD:**
```bash
python src/make_metadata_cremad.py
```

This will create `metadata.csv` files in each dataset directory (`data/RAVDESS/metadata.csv` and `data/CREMAD/metadata.csv`).

The metadata files contain columns: `utt_id`, `wav_path`, `speaker_id`, `emotion_label`.

### Step 6: Create Train/Validation/Test Splits

Generate split files that divide the data into training (70%), validation (15%), and test (15%) sets based on speaker IDs:

**For RAVDESS:**
```bash
python src/make_splits.py RAVDESS
```

**For CREMAD:**
```bash
python src/make_splits.py CREMAD
```

This creates JSON files in `configs/splits/` (`RAVDESS_splits.json` and `CREMAD_splits.json`) that map each utterance ID to its split (train/val/test).

**Note:** The splits are speaker-independent, meaning all utterances from a speaker are in the same split to prevent data leakage.

### Step 7: Verify Configuration Files

Ensure the following configuration files exist and are properly set up:

- `configs/hubert_base.json` - Model training configuration
- `configs/label_maps/dataset_to_common_7.json` - Maps dataset-specific labels to common 7-label set
- `configs/label_maps/common7_to_vad.json` - Maps common labels to VAD values

These files should already be present in the repository. You can review and modify `configs/hubert_base.json` if you want to adjust training hyperparameters (batch size, learning rate, epochs, etc.).

### Step 8: Verify Setup

Before training, verify your setup:

1. **Check data directories:**
   - `data/RAVDESS/` should contain actor folders with `.wav` files
   - `data/CREMAD/` should contain `.wav` files
   - Both should have `metadata.csv` files

2. **Check split files:**
   - `configs/splits/RAVDESS_splits.json` should exist (if using RAVDESS)
   - `configs/splits/CREMAD_splits.json` should exist (if using CREMAD)

3. **Check Python packages:**
   ```bash
   python -c "import torch; import transformers; import librosa; print('All packages installed successfully!')"
   ```

## Training the Model

Once setup is complete, you can train the model on your chosen dataset(s):

### Train on RAVDESS:
```bash
python src/train_hubert_cls.py --dataset RAVDESS --config configs/hubert_base.json
```

### Train on CREMAD:
```bash
python src/train_hubert_cls.py --dataset CREMAD --config configs/hubert_base.json
```

The trained model will be saved in:
- `models/RAVDESS_hubert_cls/` (for RAVDESS)
- `models/CREMAD_hubert_cls/` (for CREMAD)

Training outputs checkpoints after each epoch, and the best model (based on F1 score on validation set) is saved.

**Note:** Training can take several hours depending on your hardware. The model uses HuBERT-base which requires significant computational resources.

## Running Inference

After training (or using a pre-trained model), you can run inference on audio files:

```bash
python src/infer_speech.py <path_to_audio_file.wav>
```

This will output:
- Predicted emotion label
- Probability distribution over all emotion classes
- VAD (Valence-Arousal-Dominance) values

**Example:**
```bash
python src/infer_speech.py data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
```

## Project Structure

```
Prosody-Emotion-Classifier/
├── configs/                 # Configuration files
│   ├── hubert_base.json    # Training configuration
│   ├── label_maps/         # Label mapping files
│   └── splits/             # Train/val/test split files
├── data/                   # Dataset directories (created by user)
│   ├── RAVDESS/           # RAVDESS dataset files
│   └── CREMAD/            # CREMAD dataset files
├── models/                 # Trained models (created during training)
├── src/                    # Source code
│   ├── dataset_audio.py   # Dataset utilities
│   ├── infer_speech.py    # Inference script
│   ├── make_metadata_*.py # Metadata generation scripts
│   ├── make_splits.py     # Split generation script
│   └── train_hubert_cls.py # Training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** Ensure you've activated your virtual environment and installed all requirements.

### Issue: "FileNotFoundError: data/RAVDESS/metadata.csv"
**Solution:** Run the metadata generation scripts (Step 5) after downloading and placing datasets.

### Issue: "FileNotFoundError: configs/splits/RAVDESS_splits.json"
**Solution:** Run the split generation script (Step 6) after creating metadata files.

### Issue: "CUDA out of memory" during training
**Solution:** Reduce batch size in `configs/hubert_base.json` or use a smaller model. You can also train on CPU (slower but works).

### Issue: Audio files not found
**Solution:** Verify that:
1. Datasets are extracted correctly
2. Files are in the correct directory structure
3. File paths in metadata.csv are correct (they should be relative paths)
