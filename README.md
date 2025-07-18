# ECG-LLM: Large Language Model-Powered ECG Analysis for Cardiovascular Diagnosis

A research project applying state-of-the-art Large Language Models (LLMs) to electrocardiogram (ECG) analysis for advanced cardiovascular disease prediction.

## 🎯 Project Overview

This project explores the innovative approach of converting ECG signals into textual representations that can be processed by LLMs, enabling more interpretable and context-aware cardiac diagnostics. The research builds upon recent advances in Electrocardiogram-Language Models (ELMs) and incorporates retrieval-augmented generation (RAG) techniques for enhanced accuracy.

### Key Objectives
- Transform raw ECG signals into LLM-compatible textual/token representations
- Leverage pre-trained biomedical LLMs for advanced disease prediction
- Implement RAG with medical knowledge graphs for context-aware diagnosis
- Achieve superior performance compared to traditional ML classifiers
- Generate natural language explanations for ECG findings

## 🏗️ Repository Structure

```
ecg-llm-diagnosis/
├── README.md                      # Project overview and setup instructions
├── requirements.txt               # Python dependencies
├── setup.py                      # Package installation configuration
├── .gitignore                    # Git ignore patterns
├── .env.example                  # Environment variables template
├── LICENSE                       # Project license (MIT)
│
├── config/                       # Configuration files
│   ├── __init__.py
│   ├── config.yaml              # Main configuration
│   └── model_config.yaml        # Model-specific settings
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Raw ECG signals
│   ├── processed/               # Preprocessed data
│   ├── annotations/             # Medical labels and annotations
│   └── knowledge_graphs/        # KG data for RAG
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── preprocessing/           # Signal preprocessing modules
│   │   ├── __init__.py
│   │   ├── signal_filter.py    # ECG signal filtering
│   │   ├── segmentation.py     # Waveform segmentation
│   │   └── normalization.py    # Signal normalization
│   │
│   ├── encoding/                # Signal-to-text conversion
│   │   ├── __init__.py
│   │   ├── symbolic_encoder.py # Symbolic tokenization
│   │   ├── template_generator.py # Template-based descriptions
│   │   └── bpe_tokenizer.py    # Byte-pair encoding
│   │
│   ├── models/                  # Model implementations
│   │   ├── __init__.py
│   │   ├── base_llm.py         # Base LLM interface
│   │   ├── ecg_llm.py          # ECG-specific LLM
│   │   ├── fine_tuning.py      # Fine-tuning utilities
│   │   └── prompts.py          # Prompt templates
│   │
│   ├── rag/                     # RAG components
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py  # KG construction/querying
│   │   ├── retriever.py        # Document retrieval
│   │   └── augmentation.py     # Context augmentation
│   │
│   ├── evaluation/              # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py          # Performance metrics
│   │   └── benchmarks.py       # Benchmark datasets
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── data_loader.py      # Data loading utilities
│       ├── visualization.py    # ECG visualization
│       └── logger.py           # Logging configuration
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_encoding_experiments.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation_results.ipynb
│
├── experiments/                 # Experiment configurations
│   ├── baseline/               # Baseline experiments
│   ├── symbolic_encoding/      # Symbolic approach experiments
│   └── rag_enhanced/          # RAG experiments
│
├── scripts/                    # Standalone scripts
│   ├── download_data.py       # Dataset download script
│   ├── preprocess_ecg.py      # Batch preprocessing
│   ├── train_model.py         # Model training script
│   └── evaluate.py            # Evaluation script
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_encoding.py
│   └── test_models.py
│
└── docs/                       # Documentation
    ├── architecture.md         # System architecture
    ├── data_format.md         # Data format specifications
    ├── api_reference.md       # API documentation
    └── literature/            # Related papers and references

```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for LLM fine-tuning)
- Access to ECG datasets (PTB-XL, MIT-BIH, or custom datasets)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecg-llm-diagnosis.git
cd ecg-llm-diagnosis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Quick Start

1. **Data Preparation**:
```python
from src.preprocessing import preprocess_ecg_dataset
from src.utils import load_ecg_data

# Load and preprocess ECG data
data = load_ecg_data("path/to/ecg/data")
processed_data = preprocess_ecg_dataset(data)
```

2. **Signal-to-Text Conversion**:
```python
from src.encoding import SymbolicEncoder, TemplateGenerator

# Convert ECG to symbolic tokens
encoder = SymbolicEncoder()
tokens = encoder.encode(processed_data)

# Or generate template descriptions
generator = TemplateGenerator()
descriptions = generator.generate(processed_data)
```

3. **Model Inference**:
```python
from src.models import ECGLLM

# Initialize model
model = ECGLLM(model_name="gpt-4")

# Diagnose from ECG
diagnosis = model.diagnose(ecg_text=descriptions)
```

## 📊 Datasets

The project supports multiple ECG datasets:
- **PTB-XL**: Large multi-label ECG dataset (~21k patients)
- **MIT-BIH**: Arrhythmia database
- **PhysioNet**: Various ECG databases
- **Custom datasets**: Following the specified format

## 🔬 Methodology

### 1. Signal Processing Pipeline
- Noise filtering and baseline correction
- R-peak detection and segmentation
- Feature extraction (intervals, amplitudes, morphology)

### 2. Text Encoding Approaches
- **Symbolic Tokenization**: Convert signal segments to discrete symbols
- **Template Generation**: Rule-based or LLM-generated descriptions
- **Hybrid Approach**: Combining both methods

### 3. LLM Integration
- Prompt engineering for ECG interpretation
- Fine-tuning on ECG-text pairs
- Multi-modal alignment techniques

### 4. RAG Enhancement
- Medical knowledge graph construction
- Context retrieval from clinical guidelines
- Dynamic augmentation during inference

## 📈 Results

Performance metrics on standard benchmarks:
- **Accuracy**: Target >95% on arrhythmia classification
- **AUC**: Aim for 0.96+ following ECG-LM baselines
- **Interpretability**: Natural language explanations with clinical relevance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

Key papers informing this work:
1. Yang et al. (2025) - "ECG-LM: Understanding Electrocardiogram with a Large Language Model"
2. Xie et al. (2025) - "Signal, Image, or Symbolic: Exploring the Best Input Representation"
3. "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph"

See [docs/literature/](docs/literature/) for complete bibliography.

## 👥 Team

- **Aayush Parashar** - Software Engineer & AI Specialist
- **Prof. Ganesh Nayak** - Domain Expert & Research Advisor

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Prof. Nayak's lab for providing ECG datasets and medical expertise
- Anthropic, OpenAI for LLM access
- PhysioNet for public ECG databases

---

**Note**: This is an active research project. Results and methods are subject to ongoing refinement.