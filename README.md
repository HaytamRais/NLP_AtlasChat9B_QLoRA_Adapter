# NLP_AtlasChat9B_QLoRA_Adapter
## Moroccan NLP Story Scraper

A specialized Scrapy-based web crawler designed to extract stories and articles from `9esa.com` for Moroccan Arabic (Darija) NLP research and sentiment analysis.

## 🚀 Quick Start

To start scraping and save the results to a JSON Lines file:

```bash
scrapy crawl qesa -o moroccan_corpus.jsonl
```

---

## 🏗️ Architecture Overview

The project is built on the **Scrapy** framework, following a modular architecture that separates crawling logic, data structures, and post-processing pipelines.

### 🧩 File-by-File Deep Dive

#### 1. `qesa_spider.py` (The Engine)
Located in `NLP_SCRAPING_DAT/spiders/qesa_spider.py`, this is the core of the scraper.
- **Multi-Stage Parsing**:
    - `parse`: Scans the main archive list to find category/label links.
    - `parse_story_label`: Navigates through intermediate pages containing lists of individual chapters or stories. It includes custom pagination logic to follow "Next" links sequentially.
    - `parse_article`: The final extraction stage that pulls the title, raw text, and metadata from individual story pages.
- **Encoding Management**: Explicitly handles UTF-8 decoding to ensure Arabic characters are preserved correctly from the web response.

#### 2. `items.py` (The Data Schema)
Defines the `ArticleItem` class, which acts as a structured container for the scraped data.
- Fields include `url`, `title`, `raw_text`, `publish_date`, `category`, and placeholders for processed data like `clean_text` and `sentiment_label`.

#### 3. `pipelines.py` (The Processing Factory)
Handles data cleaning and enrichment after extraction.
- **ArabicCleaningPipeline**: Uses the `PyArabic` library to:
    - Normalize Hamzas and Ligatures.
    - Remove diacritics (Harakat) for NLP uniformity.
    - Strip punctuation and non-Arabic characters while preserving the core script.
- **SentimentClassificationPipeline**: A placeholder for integrating machine learning models. It demonstrates how to classify text into a 7-point sentiment scale (Strongly Negative to Strongly Positive).

#### 4. `settings.py` (Configuration)
Contains global settings for the bot.
- **Ethical Scraping**: Configured with `ROBOTSTXT_OBEY = True` and a `DOWNLOAD_DELAY` to avoid overwhelming the target server.
- **Pipeline Orchestration**: Defines the order in which data passes through cleaning and classification.
- **Persistence**: Sets up `JOBDIR` for crawl persistence, allowing the spider to resume if interrupted.

---

## 📘 User Manual

### Prerequisites
- Python 3.8+
- Scrapy
- PyArabic

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install scrapy pyarabic
   ```

### Execution
Run the spider using the following command:
```bash
scrapy crawl qesa -o moroccan_corpus.jsonl
```

### Output Data format
The output `moroccan_corpus.jsonl` contains one JSON object per line with the following structure:
```json
{
  "url": "https://www.9esa.com/story-url",
  "title": "Story Title",
  "raw_text": "Original scraped text...",
  "clean_text": "Cleaned Arabic text for NLP...",
  "publish_date": "2026-01-01",
  "category": "Story Category",
  "sentiment_label": "Neutral"
}
```

---

## 📊 Data Processing & Analysis Pipeline

The post-scraping workflow is managed in `Data_Cleaning.ipynb`, covering everything from raw data cleaning to advanced emotion classification.

### 1. Environment & Data Loading
- **Google Colab Integration**: Uses `google.colab` to mount Google Drive for persistent storage.
- **Corpus Loading**: Loads `moroccan_corpus.jsonl` into a Pandas DataFrame for analysis.
- **Initial Inspection**: Basic exploration of clean text and record counts.

### 2. Sentence Segmentation
- Uses regex (`re.split`) to break down long story texts into individual sentences based on delimiters like `.`, `!`, `?`, `،`, and newlines. 
- This enables more granular analysis (e.g., classifying emotion per sentence rather than per story).

### 3. Sentiment Analysis (Static & Sub-word)
- **Initial Distribution**: Calculates statistics for the pre-scraped 7-point sentiment labels.
- **NLTK Tokenization**: Standard word-level tokenization using NLTK's `punkt` resources.
- **DarijaBERT Tokenization**: implements sub-word tokenization using the **`SI2M-Lab/DarijaBERT`** specialized model, ensuring compatibility with state-of-the-art Darija NLP models.

### 4. Advanced Emotion Classification (Gemini API)
The pipeline integrates the **Google Gemini API** (`gemini-1.5-flash`) for sophisticated emotion detection:
- **7-Class System**: Categorizes Darija sentences into: `JOY`, `SADNESS`, `ANGER`, `FEAR`, `DISGUST`, `SURPRISE`, or `NEUTRAL`.
- **System Prompting**: Uses a specialized system prompt to help the model understand the nuances of Moroccan Arabic context and slang.
- **Implementation**: Sentences are processed iteratively, with robust error handling for API timeouts or service errors (`ERROR_API`).

> [!NOTE]
> The Gemini classification process is computationally intensive and was designed to handle the large corpus in segments.

---

## 🧠 Psychologist Dataset Generation (`50k_samples.ipynb`)

This notebook is designed to create a custom dataset for fine-tuning a Large Language Model (LLM) to act as an **empathetic Moroccan psychologist**, providing therapeutic responses in Darija.

### 1. Data Extraction and Preparation
- **Goal**: Extract 50,000 clean sentences from the Moroccan Arabic corpus for emotion labeling.
- **Process**: 
    - Mounts Google Drive and defines I/O paths.
    - Reads the large JSONL file in memory-efficient chunks.
    - Cleans and extracts sentences from the `clean_text` field until the 50,000-sentence quota is met.
    - Saves results to `50k_sentences.json`.

### 2. Emotion Labeling with Teacher Model (Atlas-Chat-9B)
- **Goal**: Automatically label the extracted sentences with one of seven emotions: `Joy`, `Love`, `Surprise`, `Sadness`, `Anger`, `Fear`, or `Neutral`.
- **Process**:
    - Loads **`MBZUAI-Paris/Atlas-Chat-9B`** as a "teacher" model.
    - Shuffles the 50,000 sentences deterministically.
    - Uses a `get_label` function with specialized prompting to classify emotional tone.
    - *Progress Update*: Successfully labeled **16,345 sentences** before an interruption (saved in `labeled_dataset_7emotions_shuffeled.jsonl`).

### 3. "Therapist Generator" Demonstration
- **Goal**: Demonstrate the generation of therapeutic responses for emotionally charged inputs.
- **Process**:
    - Extracts a balanced subset of sentences based on emotion quotas (Sadness, Fear, Anger, Love, Surprise).
    - Introduces a `generate_dynamic_therapy` function using Atlas-Chat-9B.
    - Generates empathetic responses in various styles: **Question**, **Advice**, or **Insight**, based on the patient's text and emotion.

### 4. Generating a Balanced Dataset for QLoRA Training
- **Goal**: Create a high-quality, balanced dataset of patient-therapist interactions.
- **Process**:
    - Defines `TARGET_LIMITS` for each emotion (e.g., up to 5,000 for negative emotions, 500-1,000 for positive/neutral).
    - Shuffles the labeled data to ensure diversity.
    - Applies the `generate_dynamic_therapy` function to the selected sentences.
    - **Output**: Constructs **`psychologist_dataset_ready.jsonl`** featuring **5,601 patient-therapist pairs** in an `instruction-input-output` format, ready for QLoRA fine-tuning.

---

## 🚀 Model Fine-Tuning (`Psychologist_v1.ipynb`)

This notebook details the fine-tuning of **Atlas-Chat-9B** to specialize as a Moroccan psychologist using the custom-built Darija dataset.

### 1. Environment Setup
- **Compatibility First**: Uninstalls conflicting packages and installs stable versions of `torch`, `bitsandbytes`, `transformers` (4.47.0), `peft` (0.14.0), and `trl` (0.13.0).
- **GPU Optimization**: Configured specifically for Gemma 2/Atlas models to ensure training stability in environments like Google Colab.

### 2. Configuration & Data Preparation
- Loads the `psychologist_dataset_ready.jsonl` dataset.
- **Split Strategy**: 90% Training / 10% Evaluation.
- **Prompt Engineering**: Implements a `formatting_prompts_func` to convert data into turn-based conversations (`<start_of_turn>`, `<end_of_turn>`), establishing roles for the User and the Psychologist.

### 3. Model Loading & LoRA Configuration
- **4-bit Quantization**: Uses `BitsAndBytesConfig` for efficient memory usage.
- **PEFT/LoRA**: Targets key attention and feed-forward modules with:
    - `r=16`, `lora_alpha=16`, `lora_dropout=0.05`.
    - Task type: `CAUSAL_LM`.

### 4. Supervised Fine-Tuning (SFT)
- **Training Params**: Uses `SFTTrainer` with a 512 max sequence length, `paged_adamw_32bit` optimizer, and specific logging/evaluation strategies.
- **Checkpointing**: Even after process interruptions, the model and tokenizer adapters are saved and backed up to Google Drive.

### 5. Interactive Testing ("Ask the Psychologist")
- **Custom Inference Loop**: A dedicated testing session loads the base model merged with the new LoRA adapters.
- **Darija Interaction**: Features an `ask_psychologist` function that provides real-time, empathetic responses in Moroccan Darija, simulating a professional therapeutic environment.

---

## ☀️ The Model: `Atlas-Psychologist-Darija-v1`

As a result of the fine-tuning process, we have developed **Atlas-Psychologist-Darija-v1**, a specialized AI agent tailored for the Moroccan cultural context.

### 🧩 Core Identity & Architecture
- **Role**: An empathetic Moroccan psychologist capable of understanding and responding in native Darija.
- **Base Model**: `MBZUAI-Paris/Atlas-Chat-9B` (Gemma 2 based).
- **Fine-Tuning Method**: 4-bit QLoRA (Quantized Low-Rank Adaptation).
- **Technical Specifications**:
    - **Rank (r)**: 16
    - **Alpha**: 16
    - **Target Modules**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

### ✨ Key Features
- **Cultural Competence**: Understands Moroccan social nuances and emotional expressions unique to Darija.
- **Therapeutic Empathy**: Trained to provide responses in three distinct styles: Questioning for depth, Actionable Advice, and Empathetic Insight.
- **Efficiency**: Saved as a lightweight LoRA adapter (~216MB), allowing for easy deployment on top of the base Atlas model.

### 📂 File Structure
```text
NLP_PROJECTAtlas-Psychologist-Darija-v1/
├── adapter_model.safetensors  # The trained LoRA weights
├── adapter_config.json        # PEFT configuration parameters
├── tokenizer_config.json      # Specialized tokenization settings
└── tokenizer.json             # full vocabulary support
```

