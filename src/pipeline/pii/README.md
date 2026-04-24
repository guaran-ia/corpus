# 📄 PII Detection Pipeline

---

## Description

This module implements a pipeline for detecting **Personally Identifiable Information (PII)** in text.

It detects the following types:

- 📧 Email  
- 📞 Phone  
- 🌐 IP address  
- 🏠 Physical address  

---

## How it works

The system uses a combined approach:

Regex → detects candidates  
+  
DataFog (smart mode) → validates semantically  
=  
final detection  

A PII is kept only if **both match (AND)**.

---

## Features

- High precision (reduces false positives)  
- Supports long texts (automatic chunking)  
- Base-0 and inclusive positions (`start`, `end`)  
- PII proportion metric (`pii_prop`)  
- Global corpus report  

---

## Structure

```
src/pipeline/pii/
├── pii_methods.py
├── pii_metrics.py
├── run_pii_metrics.py
```

---

## Execution

### 1. Activate environment

```bash
source venv/bin/activate
```

### 2. Run pipeline

```bash
python3 -m src.pipeline.pii.run_pii_metrics
```

---

## Input

data/processed/

- `.jsonl` files  
- Each line = one record  

Example:

```json
{
  "text": "content..."
}
```

---

## Output

### Processed files

```json
{
  "has_pii": true,
  "pii_prop": 0.12,
  "pii_spans": [
    {
      "type": "phone",
      "start": 10,
      "end": 20,
      "text": "+595981123456"
    }
  ]
}
```

---

### Global report

outputs/report/pii_report.json

```json
{
  "total_files": 39,
  "total_records": 1308377,
  "records_with_pii": 9490,
  "records_with_pii_percentage": 0.73,
  "pii_counts_by_type": {
    "email": 2278,
    "phone": 12774,
    "ip": 2514,
    "physical_address": 2200
  }
}
```

---

## Requirements

```bash
pip install transformers torch tqdm datafog==4.3.0
```