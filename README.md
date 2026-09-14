# Smart MCQ Solver

Adaptive AI system that ranks 5 candidate answers to a knowledge-intensive
multiple-choice question and returns the top 3, optimized for MAP@3.

## Stages
1. **Baseline** — TF-IDF similarity (`src/stage1_baseline.py`) — runs anywhere, CPU only.
2. **Fine-tuned DL model** — DeBERTa-v3 multiple-choice classifier — needs GPU, run on Kaggle/Colab.
3. **Retrieval (RAG)** — Wikipedia passage retrieval for factual grounding.
4. **GenAI reasoning** — LLM (via API) scores options using retrieved context.
5. **Ensemble** — combine stage 2 + 4 scores.

## Data
Not committed to git (see `.gitignore`). Place `train.csv`, `test.csv`,
`sample_submission.csv` in `data/`, OR point `MCQ_DATA_DIR` env var
at wherever they live in each environment (see below).

## Running in each environment

### Local (LMDE)
```bash
git clone <your-repo-url>
cd mcq-solver
pip install -r requirements.txt
# put csvs in data/
python src/stage1_baseline.py
```

### Kaggle Notebook
- Attach the competition dataset directly (Add Data -> competition name);
  Kaggle mounts it at `/kaggle/input/<competition-name>/`.
- At the top of the notebook:
```python
import os
os.environ["MCQ_DATA_DIR"] = "/kaggle/input/<competition-name>"
!git clone https://github.com/<you>/mcq-solver.git
import sys; sys.path.append("mcq-solver/src")
```
- Enable GPU: Settings -> Accelerator -> GPU T4 x2 (free).

### Google Colab
```python
!git clone https://github.com/<you>/mcq-solver.git
%cd mcq-solver
!pip install -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')  # for persistence across sessions
import os
os.environ["MCQ_DATA_DIR"] = "/content/drive/MyDrive/mcq-solver-data"
```
- Runtime -> Change runtime type -> GPU T4 (free tier).

### Hugging Face Hub (model storage)
Colab/Kaggle sessions get wiped, so trained weights should be pushed to HF
Hub instead of kept only in the notebook:
```python
from huggingface_hub import login, HfApi
login()  # paste token from huggingface.co/settings/tokens
model.push_to_hub("your-username/mcq-deberta-v3")
tokenizer.push_to_hub("your-username/mcq-deberta-v3")
```
Then load from anywhere (local/Colab/Kaggle) with:
```python
from transformers import AutoModelForMultipleChoice
model = AutoModelForMultipleChoice.from_pretrained("your-username/mcq-deberta-v3")
```

## Results log
| Stage | Method | Val MAP@3 |
|---|---|---|
| 1 | TF-IDF similarity | 0.3204 |
| 1 | Random baseline (reference) | 0.3667 |
| 2 | DeBERTa-v3 fine-tuned | TBD |
| 3+4 | RAG + LLM reasoning | TBD |
| 5 | Ensemble | TBD |
