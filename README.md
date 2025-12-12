# Small Language Model - Transformer vs RNN Comparison

A PyTorch implementation comparing Transformer and RNN (LSTM) architectures for language modeling, with an interactive GUI application for querying both models.

## Features

- **BPE Tokenization**: Byte Pair Encoding for efficient subword-level tokenization
- **Dual Architecture**: Implements both Transformer and RNN (LSTM) models
- **Interactive GUI**: ChatGPT-like interface to compare model responses side-by-side
- **REST API Server**: FastAPI server for model inference, suitable for deployment on PaaS platforms
- **Model Persistence**: Save and load trained models for later use
- **Professional Code**: Well-documented, type-hinted, production-ready code

## Project Structure

```
.
├── slm.py              # Main training script and model definitions
├── slm_gui.py          # Interactive GUI application
├── api_server.py       # FastAPI REST API server for model inference
├── input.txt           # Training data (text corpus)
├── models/             # Saved models directory (created after training)
│   ├── transformer.pt
│   ├── rnn.pt
│   └── tokenizer.pkl
└── README.md           # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch bpeasy
```

### 2. Train Models

Train both models and save them:

```bash
python slm.py
```

This will:
- Load and preprocess `input.txt`
- Train a BPE tokenizer
- Train Transformer and RNN models
- Save models to `models/` directory
- Display performance comparison

### 3. Run Interactive GUI

Launch the GUI application:

```bash
python slm_gui.py
```

The GUI allows you to:
- Enter prompts and query both models
- Compare Transformer vs RNN responses side-by-side
- See real-time generation from both architectures

### 4. Run API Server

Start the FastAPI server for REST API access:

```bash
python api_server.py
```

The server will:
- Load models on startup
- Start on `http://0.0.0.0:8000` (or port specified by `PORT` environment variable)
- Provide REST endpoints for model inference
- Support CORS for frontend integration

## Usage

### Training Script (`slm.py`)

```bash
python slm.py
```

**What it does:**
- Trains BPE tokenizer on your text corpus
- Trains both Transformer and RNN models
- Saves models for later use
- Displays training metrics and comparison

### GUI Application (`slm_gui.py`)

```bash
python slm_gui.py
```

**Features:**
- Clean, modern interface
- Side-by-side comparison of model outputs
- Real-time generation
- Status indicators

### API Server (`api_server.py`)

```bash
python api_server.py
```

**Features:**
- FastAPI REST API server
- Health check endpoints
- Model inference endpoint
- CORS support for frontend integration
- Suitable for PaaS deployment (e.g., Railway, Render, Fly.io)

**API Endpoints:**

- `GET /` - Health check endpoint
- `GET /health` - Health check with model status
- `POST /generate` - Generate responses from both models

**Example Request:**

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI", "max_tokens": 150}'
```

**Example Response:**

```json
{
  "transformer": "The future of AI is bright...",
  "rnn": "The future of AI will be..."
}
```

**Environment Variables:**
- `PORT` - Server port (default: 8000)
- `ALLOWED_ORIGIN` - CORS allowed origins (default: "*", or comma-separated list)

## Model Architectures

### Transformer
- Multi-head self-attention mechanism
- Positional embeddings
- Pre-norm architecture with residual connections
- Stacked transformer blocks

### RNN (LSTM)
- Multi-layer LSTM architecture
- Sequential processing with hidden state
- Recurrent connections for context

## Hyperparameters

Default configuration (can be modified in `slm.py`):
- **Batch Size**: 64
- **Block Size**: 128 (context window)
- **Embedding Dimension**: 256
- **Attention Heads**: 4
- **Layers**: 6
- **Dropout**: 0.2
- **Vocabulary Size**: 10,000
- **Learning Rate**: 3e-4

## Requirements

- Python 3.7+
- PyTorch
- bpeasy
- FastAPI
- uvicorn
- pydantic
- tkinter (usually included with Python, for GUI only)

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- Models are saved after training for reuse
- GUI loads models asynchronously to avoid blocking
- Generation happens in separate threads for responsive UI
- Models must be trained before using the GUI

## License

This project is for educational and portfolio purposes.

