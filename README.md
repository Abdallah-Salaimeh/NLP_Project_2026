NLP_Project_2026

This is an NLP Project that has been assigned to us by Dr. Mohammad Al-Azza in PSUT

‎
‎‎

📌 Project Overview
This project focuses on Arabic Abstractive Text Summarization — meaning we take Arabic news articles and train a model to generate human-like summaries automatically.

The model we are using is AraBART (moussaKam/AraBART), a transformer-based seq2seq model built specifically for Arabic NLP tasks.

‎
‎ 
‎

📁 Project Structure

NLP_Project_2026/

│

├── DataSets/                   # Raw datasets (6 datasets combined and manipulate to become 3 datasets -Train||Validate||Test-)

├── NoteBooks/              # Jupyter notebooks

│   ├── preprocessing.ipynb # Cleaning + tokenization

│   └── training.ipynb      # Model training (WIP) (NOT DONE YET)

├── environment.yml         # Conda environment file

└── README.md

‎

‎

‎

⚙️ Setting Up the Environment
‎

We use a conda environment to make sure everyone on the team runs the exact same packages and versions. No "it works on my machine" nonsense.

‎

Step 1 — Install Anaconda or Miniconda

If you don't have conda installed, download it from:

·	Anaconda: https://www.anaconda.com/download

·	Miniconda (lighter): https://docs.conda.io/en/latest/miniconda.html

‎

Step 2 — Create the environment from the file

Open your terminal, navigate to the project folder, and run:

conda env create -f environment.yml



This reads the environment.yml file and installs everything automatically.
‎

‎

Step 3 — Activate the environment

conda activate nlp_project



You need to do this every time you open a new terminal before working on the project.
‎

‎

Step 4 — Open Jupyter

jupyter notebook




‎

‎

‎


📦 Main Dependencies

Package	Purpose

transformers	Load AraBART model and tokenizer

sentencepiece	Required by AraBART tokenizer

protobuf	Required by AraBART tokenizer

torch	Deep learning backend

pandas	Data manipulation

scikit-learn	Train/val/test split

re	Text cleaning with regex


‎ 




‎ 
‎ 

‎

‎
‎
‎ 
🧹 Preprocessing Pipeline

The preprocessing is done in notebooks/preprocessing.ipynb and follows this order:
‎

‎

1.	Cleaning — removes noise from text while preserving meaning

o	Extra whitespace

o	Quoted phrases (keeps the word, removes the quotes)

o	French quotes « »

o	Latin characters

o	Standalone Arabic letters (signatures/initials)

o	Invisible unicode characters (\u200f, \xa0, etc.)

o	Newlines
‎

‎

2.	Train / Validation / Test Split — 80% / 10% / 10%
‎

‎

3.	Tokenization — converts Arabic text to token IDs using AraBART tokenizer

o	Text: max_length=1024

o	Summary: max_length=256

⚠️ Note: Punctuation (. ، ؟) is intentionally kept as it helps the model understand sentence structure during summarization.


‎

‎

‎


👥 Team

·	Built as part of the NLP course at PSUT by Abdallah Salimeh || Faris Samarah || Bassam Jaza'eri under Dr. Mohammad Al-Azza 



