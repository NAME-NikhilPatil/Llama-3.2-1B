# Llama-3.2-1B Python Code Generation Fine-tuning

This project fine-tunes Meta's Llama-3.2-1B-Instruct model for Python code generation tasks using Parameter Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) and 4-bit quantization for efficient training on consumer GPUs.

## 🚀 Project Overview

The goal of this project is to create a specialized Python code generation model by fine-tuning the lightweight Llama-3.2-1B model on a curated dataset of Python programming instructions. The model learns to understand natural language descriptions of programming tasks and generate appropriate Python code solutions.

### Key Features

- **Efficient Fine-tuning**: Uses LoRA for parameter-efficient training
- **Memory Optimized**: 4-bit quantization with BitsAndBytes for reduced VRAM usage
- **Instruction Following**: Trained on Alpaca-formatted instruction-following dataset
- **Colab Ready**: Optimized for Google Colab with GPU acceleration
- **Production Ready**: Easy deployment and inference pipeline

## 📊 Model Details

- **Base Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Model Size**: 1 billion parameters
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit with BitsAndBytes
- **Context Length**: Inherits from base model (128k tokens)
- **Specialization**: Python code generation and programming assistance

## 🎯 Dataset

The model is trained on the `iamtarun/python_code_instructions_18k_alpaca` dataset:

- **Total Samples**: 18,612 programming instructions
- **Training Split**: 16,800 samples (90%)
- **Validation Split**: 1,812 samples (10%)
- **Format**: Alpaca instruction format with input/output pairs
- **Content**: Python programming tasks, algorithms, and code examples

### Sample Data Format

```json
{
  "instruction": "Create a function to calculate the sum of a sequence of integers.",
  "input": "[1, 2, 3, 4, 5]",
  "output": "# Python code\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum"
}
```

## 🛠️ Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ VRAM for training
- Google Colab Pro (recommended for training)

### Dependencies

Create a `requirements.txt` file:

```txt
datasets>=4.0.0
trl>=0.19.0
peft>=0.16.0
bitsandbytes>=0.46.0
torch>=2.0.0
transformers>=4.51.0
accelerate>=1.4.0
huggingface-hub
```

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NAME-NikhilPatil/Llama-3.2-1B.git
   cd Llama-3.2-1B
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up HuggingFace authentication**:
   ```python
   from huggingface_hub import login
   login(token="your_hf_token_here")
   ```

## 🚀 Usage

### Training

1. **Open the Jupyter notebook**: `Llama3_2 (1).ipynb`

2. **Configure your settings**:
   - Update HuggingFace token in `config.json`
   - Adjust training parameters as needed
   - Set output directory for model checkpoints

3. **Run the training pipeline**:
   - Execute all cells in sequence
   - Monitor training progress and loss metrics
   - Checkpoints are saved every 15 steps by default

### Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, "path/to/your/fine-tuned/model")

# Format your prompt
def format_instruction(instruction, input_text=""):
    if input_text:
        prompt = f"""<|start_header_id|>user<|end_header_id|>

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        prompt = f"""<|start_header_id|>user<|end_header_id|>

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

# Generate code
instruction = "Write a Python function to find the factorial of a number"
prompt = format_instruction(instruction)

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## ⚙️ Training Configuration

### LoRA Configuration
- **Alpha**: 16
- **Rank (r)**: 16
- **Dropout**: 0.05
- **Target Modules**: `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`

### Quantization Settings
- **Load in 4-bit**: True
- **Quantization Type**: NF4
- **Compute dtype**: bfloat16
- **Double Quantization**: True

### Training Parameters
- **Checkpoint Frequency**: Every 15 steps
- **Output Directory**: `/content/drive/MyDrive/colab_training/llama32-python-save15steps`
- **Evaluation Strategy**: Steps-based
- **Save Strategy**: Steps-based

## 🔧 Customization

### Modifying Training Data

To use your own dataset:

1. Format your data in Alpaca style
2. Update the `gen_train_input()` and `gen_val_input()` functions
3. Adjust the dataset loading logic in the notebook

### Adjusting Model Parameters

Key parameters you can modify:
- LoRA rank and alpha for adaptation strength
- Learning rate and batch size
- Maximum sequence length
- Training epochs and evaluation frequency

## 📁 Project Structure

```
Llama-3.2-1B/
├── Llama3_2 (1).ipynb          # Main training notebook
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── config.json                  # Configuration file (HF token)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama-3.2-1B base model
- **HuggingFace** for the transformers ecosystem and model hosting
- **Microsoft** for the PEFT library and LoRA implementation
- **Tim Dettmers** for BitsAndBytes quantization library
- **Dataset Creator** [@iamtarun](https://huggingface.co/iamtarun) for the Python code instructions dataset

## 📚 References

- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft)

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/NAME-NikhilPatil/Llama-3.2-1B/issues) page
2. Create a new issue with detailed information
3. Join the discussion in the repository

---

**Happy Coding!** 🐍✨