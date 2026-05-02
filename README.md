# EduTwin 🎓
**LLM-Powered Digital Twin of University Students — Backend**

---

## Files

```
edutwin/
├── generate_data.py   — Generates synthetic dataset (60 students × 8 topics)
├── digital_twin.py    — Core engine + CLI entry point
├── evaluate.py        — Evaluation pipeline (precision/recall/F1/simulation)
├── requirements.txt
└── .env               — Optional config (OLLAMA_MODEL, OLLAMA_BASE_URL)
```

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama  →  https://ollama.com/download

# 3. Pull the model (one-time, ~4.7 GB)
ollama pull llama3

# 4. Generate the dataset
python generate_data.py
```

> Ollama starts automatically after install. If it ever stops, run `ollama serve`.

---

## Usage

### Full pipeline for one student
```bash
python digital_twin.py student 5
python digital_twin.py student 5 --topic probability
python digital_twin.py student 5 --save-charts     # saves PNG files
python digital_twin.py student 5 --no-charts       # skip charts entirely
```

### Class-wide teacher overview
```bash
python digital_twin.py teacher
python digital_twin.py teacher --n 20              # first 20 students only
```

### Individual capabilities
```bash
# Personalized explanation
python digital_twin.py explain 3 ai_ml

# Performance prediction
python digital_twin.py predict 3 probability

# Intervention simulations
python digital_twin.py simulate 7

# Study plan
python digital_twin.py plan 7 --days 10

# Exam answer simulation
python digital_twin.py exam 3 ai_ml "What is gradient descent?"
```

### Evaluation
```bash
python evaluate.py
# → prints metrics to terminal
# → saves evaluation_results.json
```

---

## Optional: choose a different Llama model

Create a `.env` file:
```
OLLAMA_MODEL=llama3:8b      # default — fast
OLLAMA_MODEL=llama3:70b     # best quality (needs ~40 GB RAM)
OLLAMA_MODEL=llama3.1:8b    # newer version
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `[ERROR] Ollama is not running` | Run `ollama serve` in a separate terminal |
| `[ERROR] Ollama timed out` | Add `OLLAMA_MODEL=llama3:8b` to `.env` |
| `FileNotFoundError: enhanced_students_dataset.csv` | Run `python generate_data.py` first |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
