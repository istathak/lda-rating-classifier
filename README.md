# Intro

Everything is done through `main.ipynb` it should be a simple run all to load data, train/tune LDA and get regression results.
The last cell in the notebook will be the baseline run (aka no LDA).

to set up follow all steps below

Oh also this is KEY. in the main notebook there is a `REGENERATE_DATA` variable, if you havent loaded the data before you have to set that to True.


### 1. Clone the repository

```bash
git clone https://github.com/istathak/lda-rating-classifier

```

### 2. Create and activate a virtual environment

```bash
python3 -m venv env
source env/bin/activate         # On macOS/Linux

# On Windows (Command Prompt):
# .\env\Scripts\activate

# On Windows (PowerShell):
# .\env\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy's English language model

> ⚠️ Required step! `en_core_web_sm` is not included by default.

```bash
python -m spacy download en_core_web_sm
```





