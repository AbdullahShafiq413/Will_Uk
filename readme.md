# Will PDF Extractor (Gemini-only)

A tiny FastAPI service that ingests **PDFs of will-like documents**, extracts text using native PDF text or OCR (for scans/handwriting/phone photos), and then asks **Google Gemini** to return a **strict JSON** payload that fits a predefined schema.

> Core logic lives in `extractor.py`; the HTTP API is in `main.py`. Dependencies are listed in `requirements.txt`.

---

## What it does (at a glance)

- Reads PDFs page-by-page with **PyMuPDF**.
- If a page has little/no extractable text, it switches to **Tesseract OCR** with heavy **OpenCV** preprocessing (deskew, de-shadow, CLAHE, adaptive thresholding, denoise, sharpen).
- Stitches page text together and sends it to **Gemini (1.5 Flash → 1.5 Pro fallback)** to extract a **fixed schema** (name, address, assets, funeral preferences, gifts, etc.).
- Exposes a minimal HTTP API with **/health** and **/extract** endpoints.

---

## Project layout

```
.
├── main.py          # FastAPI app & endpoints
├── extractor.py     # OCR + preprocessing + Gemini-only field extraction
└── requirements.txt # Python dependencies
```

Sources: extractor logic fileciteturn0file0 • FastAPI app fileciteturn0file1 • dependencies fileciteturn0file2

---

## Requirements

- Python 3.10+ (recommended)
- **Tesseract** installed on the system (CLI on PATH) for OCR
- Poppler is **not** required (uses PyMuPDF instead)
- A **Google API key** for Gemini models

Python packages are pinned in `requirements.txt`. fileciteturn0file2

---

## Environment variables

Create a `.env` file in the project root and set:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

`/health` exposes whether the key is present (`gemini_key_present: true/false`). fileciteturn0file1

---

## Setup

```bash
# 1) Create & activate a virtualenv (example for bash)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Add your .env with GOOGLE_API_KEY
echo "GOOGLE_API_KEY=..." > .env

# 4) Run the API
uvicorn main:app
```
The service will start at `http://127.0.0.1:8000`.then go to http://127.0.0.1:8000/docs
then have to choose post/extract and click on try it out.

---

## API

### `GET /health`
A simple health check.
**Response:**
```json
{ "ok": true, "gemini_key_present": true }
```


---

### `POST /extract` (multipart/form-data)

Upload a PDF and choose OCR/processing options.

**Form fields** (all except `file` are optional, with defaults):
- `file` (**required**): the PDF to process (must end with `.pdf`)
- `lang` (default: `"eng"`): Tesseract language code(s) for OCR 
- `dpi` (default: `300`): render DPI for OCR (auto-bumped to >= 400 for phone scans) 
- `ocr_psm` (default: `3`): Tesseract page segmentation mode (the code tries multiple variants internally) 
- `force_ocr` (default: `false`): force OCR on every page regardless of native text 
- `ocr_on_empty_only` (default: `true`): only OCR pages with little/no native text 
- `max_pages` should be empty
- `include_text_preview` (default: `false`): include first 1000 chars and meta (pages, OCR count) in response


---

## How it works 

### 1) PDF → Text
- Extract native text per page via PyMuPDF. If the page text is < ~10 chars, mark for OCR. fileciteturn0file0
- For OCR pages: render at **>= 400 DPI**, convert to PIL image. fileciteturn0file0

### 2) Robust OCR for phone scans
Preprocessing chain (OpenCV): **deskew (OSD)** → **grayscale + CLAHE** → **de-shadow** → **adaptive threshold** → **morph open** → **unsharp mask**. Then run multiple Tesseract strategies (several PSMs on original & preprocessed images) and pick the **longest text** result. fileciteturn0file0

### 3) Gemini-only field extraction
Concatenate page texts and send to Gemini with a strict schema prompt. If the model returns non‑JSON, the code asks it to **repair** to valid JSON. For long docs (> 120k chars) it **chunks**, extracts per chunk, and then **merges** results. Models tried in order: `gemini-1.5-flash`, then `gemini-1.5-pro`. Requires `GOOGLE_API_KEY`. fileciteturn0file0

---


