# 🧠 NLP_Project_2026

This is an NLP Project that has been assigned to us by Dr. Mohammad Al-Azza in PSUT

---

# 📌 Project Overview

This project focuses on **Arabic Abstractive Text Summarization** — meaning we take Arabic news articles and train a model to generate human-like summaries automatically.

The model we are using is **AraBART** (`moussaKam/AraBART`), a transformer-based seq2seq model built specifically for Arabic NLP tasks.

---

# 📁 Project Structure
```
NLP_Project_2026/
│
├── DataSets/                       # Raw datasets (6 datasets combined and split → Train || Validate || Test)
├── NoteBooks/                      # Jupyter notebooks
│   ├── PreProcessing_NLP.ipynb     # Cleaning + tokenization
│   └── training.ipynb              # Model training (WIP)
├── environment.yml                 # Conda environment file
└── README.md
```

---

# ⚙️ Setting Up the Environment

We use a conda environment to make sure everyone on the team runs the exact same packages and versions. No "it works on my machine" nonsense.

‎

**Step 1 — Install Anaconda or Miniconda**

If you don't have conda installed, download it from:
- **Anaconda**: https://www.anaconda.com/download
- **Miniconda** (lighter): https://docs.conda.io/en/latest/miniconda.html

‎

**Step 2 — Create the environment from the file**

Open your terminal, navigate to the project folder, and run:
```bash
conda env create -f environment.yml
```

This reads the `environment.yml` file and installs everything automatically.

‎

**Step 3 — Activate the environment**
```bash
conda activate nlp_project
```

You need to do this every time you open a new terminal before working on the project.

‎

**Step 4 — Open Jupyter**
```bash
jupyter notebook
```

---

# 📦 Main Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Load AraBART model and tokenizer |
| `sentencepiece` | Required by AraBART tokenizer |
| `protobuf` | Required by AraBART tokenizer |
| `torch` | Deep learning backend |
| `pandas` | Data manipulation |
| `scikit-learn` | Train/val/test split |
| `re` | Text cleaning with regex |

---

# 🧹 Preprocessing Pipeline

The preprocessing is done in `notebooks/preprocessing.ipynb` and follows this order:

‎

**1. Cleaning** — removes noise from text while preserving meaning
- Extra whitespace
- Quoted phrases (keeps the word, removes the quotes)
- French quotes `« »`
- Latin characters
- Standalone Arabic letters (signatures/initials)
- Invisible unicode characters (`\u200f`, `\xa0`, etc.)
- Newlines

‎

**2. Train / Validation / Test Split** — 80% / 10% / 10%

‎

**3. Tokenization** — converts Arabic text to token IDs using AraBART tokenizer
- Text: `max_length=1024`
- Summary: `max_length=256`

> ⚠️ **Note**: Punctuation (`. ، ؟`) is intentionally kept as it helps the model understand sentence structure during summarization.

---

# 📦 Datasets

This project uses the following Arabic summarization datasets:

---

### 1. SumArabic
- **Description**: Arabic abstractive summarization dataset with ~84,764 news article–summary pairs (train/validation/test/out-of-domain splits), sourced from Arabic news websites.
- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/7kr75c9h24/1)
- **Contributor**: Mohammad Bani Almarjeh
- **License**: [MPL-2.0 (Mozilla Public License 2.0)](https://www.mozilla.org/en-US/MPL/2.0/)
- **DOI**: 10.17632/7kr75c9h24.1

---

### 2. AraSum
- **Description**: Large monolingual Arabic summarization corpus with ~49,604 article–lead summary pairs from Deutsche Welle Arabic. Also includes AraRLHF (1,746 samples) and AraDPO (2,309 samples) for preference-based fine-tuning.
- **Source**: [GitHub – ppke-nlpg/AraSum](https://github.com/ppke-nlpg/AraSum)
- **License**: No explicit license specified — contact authors before redistribution.
- **Citation**:
```
@inproceedings{kahla-etal-2021-cross,
    title = "Cross-lingual Fine-tuning for Abstractive Arabic Text Summarization",
    author = "Kahla, Mram and Yang, Zijian Győző and Novák, Attila",
    booktitle = "Proceedings of RANLP 2021",
    year = "2021",
    url = "https://aclanthology.org/2021.ranlp-1.74"
}
```

---

### 3. Egyptian Arabic Text Summarization
- **Description**: ~3,690 text–summary pairs in Egyptian Arabic dialect, with topic categorization. Designed for dialectal summarization evaluation.
- **Source**: [Hugging Face – Omar-youssef/Egyptian-text-summarization](https://huggingface.co/datasets/Omar-youssef/Egyptian-text-summarization)
- **License**: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

### 4. Arabic Text Summarization (Kaggle)
- **Description**: Arabic summarization dataset available on Kaggle.
- **Source**: [Kaggle – abdalrahmanshahrour/arabicsummarization](https://www.kaggle.com/datasets/abdalrahmanshahrour/arabicsummarization)
- **Original Data Source**: [Arabic News Texts Corpus – muhammedfathi](https://www.kaggle.com/datasets/muhammedfathi/arabic-news-texts-corpus)
- **License**: ⚠️ Not explicitly listed — verify on the dataset page before use. Kaggle datasets require accepting the platform's terms of service.

---

> **Note**: Always verify dataset licenses and usage terms before training or publishing results. Some datasets may require citation or restrict commercial use.

---

# 👥 Team

Built as part of the **NLP course at PSUT** by:

**Abdallah Salimeh** || **Faris Samarah** || **Bassam Jaza'eri**

under **Dr. Mohammad Al-Azza**
