"""
Small Language Model: A PyTorch implementation comparing Transformer and RNN architectures.

This module implements a complete language modeling pipeline including:
- BPE (Byte Pair Encoding) tokenization
- Transformer-based language model with multi-head self-attention
- RNN-based language model using LSTM
- Training, evaluation, and text generation capabilities
"""

from typing import Dict, List, Optional, Tuple
import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import bpeasy
import time

# ============================================================================
# Hyperparameters
# ============================================================================
BATCH_SIZE = 128  # Number of sequences per batch (B)
BLOCK_SIZE = 128  # Context window length (T)
MAX_ITERS = 10000  # Maximum training iterations
EVAL_INTERVAL = 100  # Evaluation frequency
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 50  # Number of batches for loss estimation
N_EMBD = 256  # Embedding dimension (C)
N_HEAD = 4  # Number of attention heads (H)
N_LAYER = 6  # Number of transformer/RNN layers (L)
DROPOUT = 0.2  # Dropout probability
VOCAB_SIZE = 10000  # Vocabulary size

# Reproducibility
torch.manual_seed(1337)

class Tokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer for subword-level tokenization.
    
    Implements a trainable tokenizer that learns subword units from text,
    enabling efficient handling of out-of-vocabulary words.
    """

    def __init__(self, vocab_size: int = 10000) -> None:
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size for BPE training
        """
        self.vocab_size = vocab_size
        self.encoder: Dict[str, int] = {}  # subword -> token_id mapping
        self.decoder: Dict[int, str] = {}  # token_id -> subword mapping
        self.vocab_tokens: List[str] = []  # sorted list of subwords

    def train(self, text: str) -> None:
        """
        Train BPE tokenizer on provided text.
        
        Learns subword vocabulary and builds encoder/decoder mappings.
        
        Args:
            text: Training text corpus
        """
        vocab = bpeasy.train_bpe(
            iter(text.split()),
            python_regex=r"\S+",
            max_token_length=50,
            vocab_size=self.vocab_size,
        )

        # Build encoder/decoder dictionaries
        for token, idx in vocab.items():
            try:
                token_str = (
                    token.decode("utf-8") if isinstance(token, bytes) else str(token)
                )
            except UnicodeDecodeError:
                token_str = str(token)
            self.encoder[token_str] = idx
            self.decoder[idx] = token_str
            self.vocab_tokens.append(token_str)

        # Sort by length (longest first) for greedy matching
        self.vocab_tokens.sort(key=lambda x: (len(x), x), reverse=True)

        # Add special tokens
        unk_idx = len(self.encoder)
        self.encoder["<unk>"] = unk_idx
        self.decoder[unk_idx] = "<unk>"
        
        # Add newline token if not already in vocabulary
        if "\n" not in self.encoder:
            newline_idx = len(self.encoder)
            self.encoder["\n"] = newline_idx
            self.decoder[newline_idx] = "\n"
            self.vocab_tokens.append("\n")
            # Re-sort to maintain longest-first order
            self.vocab_tokens.sort(key=lambda x: (len(x), x), reverse=True)

    def bpe(self, word: str) -> List[int]:
        """
        Apply BPE algorithm to split word into subword tokens.
        
        Uses greedy longest-match strategy to tokenize words.
        
        Args:
            word: Input word to tokenize
            
        Returns:
            List of token IDs representing the word
        """
        if word in self.encoder:
            return [self.encoder[word]]

        tokens = []
        i = 0
        while i < len(word):
            match = None
            # Greedy longest-match: find longest subword starting at position i
            for tok in self.vocab_tokens:
                if word.startswith(tok, i):
                    match = tok
                    break
            
            if match:
                tokens.append(self.encoder.get(match, self.encoder["<unk>"]))
                i += len(match)
            else:
                # Fallback to character-level tokenization
                tokens.append(self.encoder.get(word[i], self.encoder["<unk>"]))
                i += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text into sequence of token IDs, preserving newlines.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        token_ids = []
        i = 0
        
        while i < len(text):
            # Handle newlines explicitly
            if text[i] == "\n":
                if "\n" in self.encoder:
                    token_ids.append(self.encoder["\n"])
                i += 1
                continue
            
            # Find the next newline or end of text
            next_newline = text.find("\n", i)
            if next_newline == -1:
                segment = text[i:]
                end_idx = len(text)
            else:
                segment = text[i:next_newline]
                end_idx = next_newline
            
            # Tokenize the segment (words separated by spaces)
            if segment.strip():  # Only process non-empty segments
                words = segment.split()
                for j, word in enumerate(words):
                    token_ids.extend(self.bpe(word))
                    # Add space token between words (if space is in vocabulary)
                    if j < len(words) - 1 and " " in self.encoder:
                        token_ids.append(self.encoder[" "])
            
            i = end_idx
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Reconstructed text string
        """
        tokens = [self.decoder.get(tid, "<unk>") for tid in token_ids]
        return "".join(tokens)


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input-target pairs from dataset.
    
    Args:
        split: "train" or "val" to select dataset
        train_data: Training dataset tensor
        val_data: Validation dataset tensor
        
    Returns:
        Tuple of (input_batch, target_batch) tensors of shape (B, T)
    """
    data = train_data if split == "train" else val_data
    # Random starting indices for sequences
    indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # Stack sequences: inputs are [i:i+T], targets are [i+1:i+T+1]
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in indices])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in indices])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> Dict[str, float]:
    """
    Estimate average loss on train and validation sets.
    
    Args:
        model: Model to evaluate
        train_data: Training dataset
        val_data: Validation dataset
        
    Returns:
        Dictionary with "train" and "val" loss values
    """
    model.eval()
    losses_dict = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)[:2]
            losses[k] = loss.item()
        losses_dict[split] = losses.mean().item()
    model.train()
    return losses_dict


class Head(nn.Module):
    """
    Single self-attention head implementing scaled dot-product attention.
    
    Computes attention weights and aggregates values based on query-key similarity.
    """

    def __init__(self, head_size: int) -> None:
        """
        Initialize attention head.
        
        Args:
            head_size: Dimension of each attention head
        """
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)

        # Causal mask: prevents attending to future tokens
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention head.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, head_size)
        """
        _, T, C = x.shape

        # Compute Q, K, V projections
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Scaled dot-product attention
        # Attention scores: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted aggregation: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Runs multiple attention heads in parallel and combines their outputs.
    """

    def __init__(self, num_heads: int, head_size: int) -> None:
        """
        Initialize multi-head attention.
        
        Args:
            num_heads: Number of parallel attention heads
            head_size: Dimension of each attention head
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Output projection to combine heads
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two-layer MLP with ReLU activation and dropout.
    """

    def __init__(self, n_embd: int) -> None:
        """
        Initialize feed-forward network.
        
        Args:
            n_embd: Embedding dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Project back
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture.
    
    Combines self-attention (communication) and feed-forward (computation)
    with residual connections and layer normalization.
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        """
        Initialize transformer block.
        
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        # Pre-norm with residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BaseLanguageModel(nn.Module):
    """
    Base class for language models with shared components.
    
    Provides token embeddings, output projection, loss computation,
    and text generation functionality.
    """

    def __init__(self) -> None:
        """Initialize base language model components."""
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def compute_loss(
        self, logits: torch.Tensor, targets: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute cross-entropy loss between predictions and targets.
        
        Args:
            logits: Model predictions of shape (B, T, vocab_size)
            targets: Target token IDs of shape (B, T)
            
        Returns:
            Scalar loss value, or None if targets not provided
        """
        if targets is None:
            return None
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """
        Generate text by autoregressively sampling from the model.
        
        Args:
            idx: Initial context tokens of shape (B, T)
            max_tokens: Number of tokens to generate
            
        Returns:
            Generated token sequence of shape (B, T + max_tokens)
        """
        for _ in range(max_tokens):
            # Use full context for transformers, single token for RNNs
            idx_cond = (
                idx[:, -BLOCK_SIZE:] if hasattr(self, "blocks") else idx[:, -1:]
            )
            logits, _ = self(idx_cond)
            # Sample from last position
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TransformerLanguageModel(BaseLanguageModel):
    """
    Transformer-based language model using multi-head self-attention.
    
    Implements a decoder-only transformer architecture with positional
    embeddings and stacked transformer blocks.
    """

    def __init__(self) -> None:
        """Initialize transformer language model."""
        super().__init__()
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[TransformerBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        )

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer model.
        
        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices for loss computation
            
        Returns:
            Tuple of (logits, loss) where logits are (B, T, vocab_size)
        """
        _, T = idx.shape
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, self.compute_loss(logits, targets)


class RNNLanguageModel(BaseLanguageModel):
    """
    RNN-based language model using LSTM architecture.
    
    Processes sequences sequentially using recurrent connections
    to maintain hidden state across tokens.
    """

    def __init__(self) -> None:
        """Initialize RNN language model."""
        super().__init__()
        self.lstm = nn.LSTM(
            N_EMBD, N_EMBD, N_LAYER, batch_first=True, dropout=DROPOUT
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through RNN model.
        
        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices for loss computation
            hidden: Optional LSTM hidden state (h, c)
            
        Returns:
            Tuple of (logits, loss, hidden_state)
        """
        B, _ = idx.shape
        tok_emb = self.token_embedding_table(idx)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = (
                torch.zeros(N_LAYER, B, N_EMBD, device=idx.device),
                torch.zeros(N_LAYER, B, N_EMBD, device=idx.device),
            )

        lstm_out, hidden = self.lstm(tok_emb, hidden)
        x = self.ln_f(lstm_out)
        logits = self.lm_head(x)
        return logits, self.compute_loss(logits, targets), hidden

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """
        Generate text using RNN with maintained hidden state.
        
        Args:
            idx: Initial context tokens of shape (B, T)
            max_tokens: Number of tokens to generate
            
        Returns:
            Generated token sequence of shape (B, T + max_tokens)
        """
        hidden = None
        for _ in range(max_tokens):
            idx_cond = idx[:, -1:]  # Process one token at a time
            logits, _, hidden = self(idx_cond, hidden=hidden)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train(
    model: nn.Module,
    model_name: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> Tuple[nn.Module, List[float], List[float], float]:
    """
    Train a language model with early stopping based on validation loss.
    
    Args:
        model: Model to train
        model_name: Name for logging purposes
        train_data: Training dataset
        val_data: Validation dataset
        
    Returns:
        Tuple of (trained_model, train_losses, val_losses, training_time)
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    best_iteration = 0
    patience_counter = 0
    PATIENCE = 3  # Early stopping patience
    MIN_DELTA = 1e-3  # Minimum improvement threshold

    print(f"\nTraining {model_name}...")
    start_time = time.time()

    for iteration in range(MAX_ITERS):
        # Periodic evaluation
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            
            # Early stopping check
            if losses["val"] < best_val_loss - MIN_DELTA:
                best_val_loss = losses["val"]
                best_iteration = iteration
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at iteration {iteration}")
                    print(
                        f"Restoring best model from iteration {best_iteration} "
                        f"(val loss: {best_val_loss:.4f})"
                    )
                    # Restore best model weights
                    model.load_state_dict(best_model_state)
                    break

            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            print(
                f"step {iteration}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        # Training step
        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)[:2]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")

    return model, train_losses, val_losses, training_time


def save_models(
    transformer: nn.Module,
    rnn: nn.Module,
    tokenizer: Tokenizer,
    save_dir: str = "models",
) -> None:
    """
    Save trained models and tokenizer to disk.
    
    Args:
        transformer: Trained transformer model
        rnn: Trained RNN model
        tokenizer: Trained tokenizer
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save models
    torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer.pt"))
    torch.save(rnn.state_dict(), os.path.join(save_dir, "rnn.pt"))
    
    # Save tokenizer
    with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    
    print(f"\nModels saved to {save_dir}/")


def load_models(
    save_dir: str = "models",
) -> Tuple[nn.Module, nn.Module, Tokenizer]:
    """
    Load trained models and tokenizer from disk.
    
    Args:
        save_dir: Directory containing saved models
        
    Returns:
        Tuple of (transformer_model, rnn_model, tokenizer)
    """
    # Load tokenizer with custom unpickler to handle module path issues
    # This fixes the case where tokenizer was saved from __main__ (slm.py)
    # but is being loaded from a different __main__ (slm_gui.py)
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Redirect Tokenizer class lookup to current module
            if name == "Tokenizer":
                return Tokenizer
            # For other classes, try normal import
            return super().find_class(module, name)
    
    with open(os.path.join(save_dir, "tokenizer.pkl"), "rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
    
    # Initialize models
    transformer = TransformerLanguageModel()
    rnn = RNNLanguageModel()
    
    # Load state dicts
    transformer.load_state_dict(
        torch.load(os.path.join(save_dir, "transformer.pt"), map_location=DEVICE)
    )
    rnn.load_state_dict(
        torch.load(os.path.join(save_dir, "rnn.pt"), map_location=DEVICE)
    )
    
    transformer.eval()
    rnn.eval()
    transformer = transformer.to(DEVICE)
    rnn = rnn.to(DEVICE)
    
    print(f"Models loaded from {save_dir}/")
    return transformer, rnn, tokenizer


def generate_response(
    prompt: str,
    transformer: nn.Module,
    rnn: nn.Module,
    tokenizer: Tokenizer,
    max_tokens: int = 150,
) -> Tuple[str, str]:
    """
    Generate responses from both models for a given prompt.
    
    Args:
        prompt: Input text prompt
        transformer: Trained transformer model
        rnn: Trained RNN model
        tokenizer: Tokenizer for encoding/decoding
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (transformer_response, rnn_response)
    """
    # Encode prompt
    context = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE
    )
    
    # Generate with transformer
    with torch.no_grad():
        transformer_gen = transformer.generate(context, max_tokens=max_tokens)
        transformer_text = tokenizer.decode(transformer_gen[0].tolist())
    
    # Generate with RNN
    with torch.no_grad():
        rnn_gen = rnn.generate(context, max_tokens=max_tokens)
        rnn_text = tokenizer.decode(rnn_gen[0].tolist())
    
    return transformer_text, rnn_text


def compare(
    transformer: nn.Module,
    rnn: nn.Module,
    tokenizer: Tokenizer,
) -> None:
    """
    Compare text generation quality between transformer and RNN models.
    
    Args:
        transformer: Trained transformer model
        rnn: Trained RNN model
        tokenizer: Tokenizer for encoding/decoding
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON - Text Generation")
    print("=" * 60)
    
    test_prompt = "Where is"
    print(f"\nTest prompt: '{test_prompt}'")

    transformer_text, rnn_text = generate_response(test_prompt, transformer, rnn, tokenizer)
    
    print(f"\nTransformer Generated Text:")
    print("-" * 60)
    print(transformer_text)
    print(f"\nRNN Generated Text:")
    print("-" * 60)
    print(rnn_text)


def main() -> None:
    """
    Main execution pipeline: data loading, tokenization, training, and evaluation.
    
    Orchestrates the complete workflow:
    1. Load and preprocess text data
    2. Train BPE tokenizer
    3. Initialize and train both Transformer and RNN models
    4. Compare model performance and generation quality
    """
    # Load text data
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("input.txt not found. Please provide a text file for training.")

    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(text)

    # Encode text into token sequences
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # Ensure minimum dataset size
    if len(data) < BLOCK_SIZE * 10:
        data = data.repeat(10)

    print(f"Data length: {len(data)} tokens")

    # Train/validation split (90/10)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(
        f"Vocabulary size: {len(tokenizer.encoder)}, "
        f"Block size: {BLOCK_SIZE}, "
        f"Embedding dimension: {N_EMBD}"
    )

    # Initialize and train models
    models = [TransformerLanguageModel(), RNNLanguageModel()]
    model_names = ["Transformer", "RNN"]
    trained_models = []
    all_losses = []
    training_times = []

    for model, name in zip(models, model_names):
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n{name} parameters: {num_params:.2f}M")
        
        # Train model
        trained_model, train_loss, val_loss, elapsed_time = train(
            model, name, train_data, val_data
        )
        trained_models.append(trained_model)
        all_losses.append({"train": train_loss, "val": val_loss})
        training_times.append(elapsed_time)

    # Save trained models
    save_models(*trained_models, tokenizer)

    # Compare model generation quality
    compare(*trained_models, tokenizer)

    # Print final performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    for name, losses, elapsed_time in zip(model_names, all_losses, training_times):
        final_train_loss = losses["train"][-1] if losses["train"] else float("inf")
        final_val_loss = losses["val"][-1] if losses["val"] else float("inf")
        print(
            f"{name:12s}: Train Loss {final_train_loss:.4f} | "
            f"Val Loss {final_val_loss:.4f} | "
            f"Time {elapsed_time:.2f}s"
        )


if __name__ == "__main__":
    main()
