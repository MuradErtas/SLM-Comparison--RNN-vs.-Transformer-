"""
Interactive GUI application for querying Transformer and RNN language models.

This application provides a ChatGPT-like interface to compare responses
from both Transformer and RNN models side by side.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from slm import load_models, generate_response, DEVICE, Tokenizer

class LanguageModelGUI:
    """GUI application for interactive language model comparison."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Language Model Comparison - Transformer vs RNN")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load models
        self.transformer = None
        self.rnn = None
        self.tokenizer = None
        self.models_loaded = False
        self.models_dir = "models"  # Default models directory
        
        self.setup_ui()
        self.load_models_async()
    
    def setup_ui(self) -> None:
        """Set up the user interface components."""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="Language Model Comparison",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#2c3e50"
        )
        title_label.pack(pady=15)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Loading models...",
            font=("Arial", 10),
            bg="#ecf0f1",
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input section
        input_frame = tk.LabelFrame(
            main_frame,
            text="Enter your prompt",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10
        )
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            height=3,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white"
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        self.input_text.bind("<Return>", lambda e: self.on_generate())
        self.input_text.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for newline
        
        # Generate button
        button_frame = tk.Frame(input_frame, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.generate_btn = tk.Button(
            button_frame,
            text="Generate Response",
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            command=self.on_generate,
            cursor="hand2",
            padx=20,
            pady=8
        )
        self.generate_btn.pack(side=tk.LEFT)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 11),
            bg="#95a5a6",
            fg="white",
            activebackground="#7f8c8d",
            command=self.clear_all,
            cursor="hand2",
            padx=15,
            pady=8
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Response section - side by side
        response_frame = tk.Frame(main_frame, bg="#f0f0f0")
        response_frame.pack(fill=tk.BOTH, expand=True)
        
        # Transformer panel
        transformer_frame = tk.LabelFrame(
            response_frame,
            text="Transformer Model",
            font=("Arial", 12, "bold"),
            bg="#e8f5e9",
            fg="#2e7d32",
            padx=10,
            pady=10
        )
        transformer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.transformer_text = scrolledtext.ScrolledText(
            transformer_frame,
            font=("Arial", 10),
            wrap=tk.WORD,
            bg="white",
            state=tk.DISABLED,
            padx=10,
            pady=10
        )
        self.transformer_text.pack(fill=tk.BOTH, expand=True)
        
        # RNN panel
        rnn_frame = tk.LabelFrame(
            response_frame,
            text="RNN Model (LSTM)",
            font=("Arial", 12, "bold"),
            bg="#fff3e0",
            fg="#e65100",
            padx=10,
            pady=10
        )
        rnn_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.rnn_text = scrolledtext.ScrolledText(
            rnn_frame,
            font=("Arial", 10),
            wrap=tk.WORD,
            bg="white",
            state=tk.DISABLED,
            padx=10,
            pady=10
        )
        self.rnn_text.pack(fill=tk.BOTH, expand=True)
    
    def load_models_async(self) -> None:
        """Load models in a separate thread to avoid blocking UI."""
        def load():
            try:
                self.transformer, self.rnn, self.tokenizer = load_models(self.models_dir)
                self.models_loaded = True
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Models loaded successfully | Device: {DEVICE}",
                    fg="green"
                ))
                self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            except FileNotFoundError as e:
                error_msg = (
                    f"Models not found. Please ensure 'models' directory exists.\n"
                    f"Run 'python slm.py' first to train and save models."
                )
                self.root.after(0, lambda: self.status_label.config(
                    text=error_msg,
                    fg="red"
                ))
                self.root.after(0, lambda: messagebox.showerror("Models Not Found", error_msg))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error loading models: {str(e)}",
                    fg="red"
                ))
                self.root.after(0, lambda: messagebox.showerror("Load Error", str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_generate(self) -> None:
        """Handle generate button click."""
        if not self.models_loaded:
            messagebox.showwarning("Models Not Ready", "Please wait for models to load.")
            return
        
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showinfo("Empty Prompt", "Please enter a prompt.")
            return
        
        # Disable button during generation
        self.generate_btn.config(state=tk.DISABLED, text="Generating...")
        self.status_label.config(text="Generating responses...", fg="blue")
        
        # Clear previous responses
        self.transformer_text.config(state=tk.NORMAL)
        self.transformer_text.delete("1.0", tk.END)
        self.transformer_text.insert("1.0", "Generating...")
        self.transformer_text.config(state=tk.DISABLED)
        
        self.rnn_text.config(state=tk.NORMAL)
        self.rnn_text.delete("1.0", tk.END)
        self.rnn_text.insert("1.0", "Generating...")
        self.rnn_text.config(state=tk.DISABLED)
        
        # Generate in separate thread
        def generate():
            try:
                transformer_response, rnn_response = generate_response(
                    prompt, self.transformer, self.rnn, self.tokenizer
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self.update_responses(
                    transformer_response, rnn_response
                ))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error generating response: {str(e)}",
                    fg="red"
                ))
                self.root.after(0, lambda: self.generate_btn.config(
                    state=tk.NORMAL, text="Generate Response"
                ))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def update_responses(self, transformer_response: str, rnn_response: str) -> None:
        """Update the response text areas."""
        # Update transformer response
        self.transformer_text.config(state=tk.NORMAL)
        self.transformer_text.delete("1.0", tk.END)
        self.transformer_text.insert("1.0", transformer_response)
        self.transformer_text.config(state=tk.DISABLED)
        
        # Update RNN response
        self.rnn_text.config(state=tk.NORMAL)
        self.rnn_text.delete("1.0", tk.END)
        self.rnn_text.insert("1.0", rnn_response)
        self.rnn_text.config(state=tk.DISABLED)
        
        # Re-enable button
        self.generate_btn.config(state=tk.NORMAL, text="Generate Response")
        self.status_label.config(text="Ready", fg="green")
    
    def clear_all(self) -> None:
        """Clear all text areas."""
        self.input_text.delete("1.0", tk.END)
        self.transformer_text.config(state=tk.NORMAL)
        self.transformer_text.delete("1.0", tk.END)
        self.transformer_text.config(state=tk.DISABLED)
        self.rnn_text.config(state=tk.NORMAL)
        self.rnn_text.delete("1.0", tk.END)
        self.rnn_text.config(state=tk.DISABLED)


def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    app = LanguageModelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

