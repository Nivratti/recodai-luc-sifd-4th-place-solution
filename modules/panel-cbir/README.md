# 🧠 Panel CBIR — Content-Based Image Retrieval for Scientific Figures

A modular, GPU-accelerated content-based image retrieval (CBIR) toolkit designed for **scientific figure panels** (microscopy, Western blots, histology, etc.).
Supports **paths**, **NumPy arrays**, and **PIL images**, with optimized batched ranking.

---

## 🚀 Features

✅ Unified API: accept image paths, NumPy arrays, or PIL images
✅ Batch-optimized embedding and ranking (GPU-accelerated)
✅ Reusable indexes (embed once, reuse many)
✅ Optional disk caching (`.npz` index)
✅ Torch backend with parallel loading (for path inputs)
✅ Clean Python + CLI interface

---

## 📦 Installation

```bash
pip install -e .
```

This installs the `panel-cbir` command-line tool and Python API.

---

## 🧩 Quick Start (Python)

### 1️⃣ Import and configure

```python
from panel_cbir import PanelCBIR, CBIRConfig

cbir = PanelCBIR(
    CBIRConfig(
        device="cuda",         # or "cpu"
        batch_size=64,
        num_workers=4,
        score_fp16=True,       # faster GPU matmul
    )
)
```

---

### 2️⃣ Build index (paths, arrays, or PIL images)

```python
import numpy as np
from PIL import Image

images = [
    "img1.png",                             # path
    np.random.randint(0, 255, (256, 256, 3), np.uint8),  # numpy array
    Image.open("img2.jpg"),                 # PIL image
]

index = cbir.build_index(images, ids=["a", "b", "c"])
```

> 💡 Embeddings are computed **once** and cached in memory or disk (`cache_path`).

---

### 3️⃣ Rank within same list

```python
rankings = cbir.rank_within_index(index, topk=5)
```

Returns a dictionary:

```python
{
  "a": [{"id": "b", "score": 0.94}, {"id": "c", "score": 0.88}],
  "b": [{"id": "a", "score": 0.94}, {"id": "c", "score": 0.79}],
  ...
}
```

---

### 4️⃣ Rank separate queries vs candidates

```python
queries = [
    "query1.png",
    np.random.rand(224, 224, 3).astype("float32"),
]

results = cbir.rank(index, queries, query_ids=["q1", "q2"], topk=5)
```

---

## ⚙️ CLI Usage

```bash
panel-cbir rank \
  --input-dir "./samples" \
  --out "runs/cbir_out" \
  --device "cuda" \
  --topk 10
```

Supports batch embedding, GPU inference, and CSV/JSON export of similarity results.

---

## ⚡ Performance Optimizations

| Optimization                | Description                                      |
| --------------------------- | ------------------------------------------------ |
| **Embed once**              | Reuse embeddings for multiple queries            |
| **Matrix multiply ranking** | Uses efficient GEMM (`E @ E.T`) instead of loops |
| **Parallel image loading**  | PyTorch `DataLoader` for path inputs             |
| **Mixed precision (fp16)**  | Optional GPU speed boost                         |
| **Chunked similarity**      | Prevents GPU OOM for large datasets              |

---

## 🧠 API Reference (summary)

| Class / Function      | Description                              |
| --------------------- | ---------------------------------------- |
| `CBIRConfig`          | Dataclass for device, batch size, etc.   |
| `PanelCBIR`           | Main interface for embedding and ranking |
| `build_index()`       | Compute embeddings (path / NumPy / PIL)  |
| `rank()`              | Rank queries vs index                    |
| `rank_within_index()` | Rank all-vs-all within one set           |
| `embed_query()`       | Get single embedding (for custom logic)  |

---

## 🗂 Example Integration in Pipeline

```python
from panel_cbir import PanelCBIR, CBIRConfig

cbir = PanelCBIR(CBIRConfig(device="cuda"))
index = cbir.build_index(panel_images, ids=panel_ids)
reuse_results = cbir.rank_within_index(index, topk=3)
```

Used in panel-reuse or copy-move detection tasks for figure-integrity analysis.

---

## 📁 Project Structure

```
panel-cbir/
├── pyproject.toml
├── README.md
└── src/
    └── panel_cbir/
        ├── __init__.py
        ├── core.py
        ├── config.py
        ├── io_utils.py
        ├── embedder.py
        ├── ranker.py
        └── legacy/   # optional old code kept for compatibility
```

---

## 🧪 Notes

* For best performance, use GPU (`cuda`).
* For ~30 images (pairwise ranking):
  → Embeds once (~0.1 s/image) and computes full 30×30 similarity instantly.

---