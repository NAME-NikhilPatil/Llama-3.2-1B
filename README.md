# Fine-Tuning Llama 3.2 for Python Code Generation

## Overview

This project demonstrates how to fine-tune the `meta-llama/Llama-3.2-1B-Instruct` model on a dataset of Python code instructions. The goal is to improve the model's ability to generate Python code based on natural language prompts.

The process includes:
- Loading and preparing a specialized dataset for code generation.
- Fine-tuning the model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
- Using 4-bit quantization to reduce memory usage, making it feasible to run on a single T4 GPU in Google Colab.
- Evaluating the fine-tuned model on the HumanEval benchmark to measure its code generation capabilities.

## Features

- **Model:** `meta-llama/Llama-3.2-1B-Instruct`
- **Dataset:** `iamtarun/python_code_instructions_18k_alpaca`
- **Fine-Tuning Technique:** PEFT with LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes`
- **Evaluation:** HumanEval benchmark using the `evaluate` library.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create a Python Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Hugging Face Access Token

To download the Llama 3.2 model, you need a Hugging Face access token.

1.  Create a `config.json` file by copying the template:
    ```bash
    cp config.json.template config.json
    ```
2.  Open `config.json` and replace `"your_hugging_face_access_token_here"` with your actual Hugging Face token. You can get a token from your [Hugging Face settings](https://huggingface.co/settings/tokens).

    ```json
    {
      "HF_ACCESS_TOKEN": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    }
    ```

## Usage

The entire process is contained within the `finetune_llama_for_code_generation.ipynb` Jupyter notebook.

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open the `finetune_llama_for_code_generation.ipynb` notebook.
3.  **Run the cells:** Execute the cells in order to download the data, prepare the model, run the fine-tuning process, and evaluate the results.

**Note:** This notebook is designed to be run in an environment with a GPU (e.g., Google Colab with a T4 GPU runtime).

### Speed-Up Knobs (Optional)

If you want faster iterations in Colab, toggle the `QUICK_RUN` flag in the data preparation cell. It reduces the train/validation sample counts and shortens the training run. You can also adjust `TRAIN_SAMPLES`, `VAL_SAMPLES`, and `SHUFFLE_BUFFER` to trade off speed and data coverage. The training cell additionally enables sequence packing with `max_seq_length=1024` to improve throughput for short examples.

## Notebook Structure

The notebook is organized into the following sections:

1.  **Setup:** Installs dependencies and mounts Google Drive for saving checkpoints.
2.  **Data Preparation:** Loads the `iamtarun/python_code_instructions_18k_alpaca` dataset and formats it for training.
3.  **Model Preparation:** Loads the `meta-llama/Llama-3.2-1B-Instruct` model, applies 4-bit quantization, and sets up the LoRA configuration.
4.  **Training:** Fine-tunes the model using the `SFTTrainer` from the TRL library.
5.  **Evaluation:** Loads the fine-tuned model and evaluates its performance on the HumanEval benchmark, calculating `pass@1` and `pass@10` scores.

## Evaluation Results

The fine-tuned model was evaluated on the HumanEval dataset, achieving the following scores:

- **pass@1:** 0.0476
- **pass@10:** 0.0915

These metrics indicate the model's ability to generate functionally correct Python code for unseen problems.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
