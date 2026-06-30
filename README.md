<div align="center">

# 🎌 智能日语词汇记忆辅助系统

## Intelligent Japanese Vocabulary Memory Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A full-stack, AI-powered vocabulary learning system that makes mastering Japanese words smarter — not harder.*

</div>

---

## 📖 Overview

The **Intelligent Japanese Vocabulary Memory Assistant** (智能日语词汇记忆辅助系统) is a Streamlit-based web application designed to help Japanese learners efficiently acquire, review, and retain vocabulary. Unlike static flashcard apps, it adapts to your learning behavior in real time — using **machine learning** to predict which words you're about to forget, **adaptive testing** to focus on your weak spots, and **rich analytics** to visualize your progress.

**Why this matters:** Most vocabulary tools show you words in a fixed order. This system observes your interactions, builds a personalized forgetting-risk model for every word, and dynamically prioritizes what needs your attention *right now*.

---

## ✨ Key Features

### 🧠 Smart Review Engine
- **Forgetting Risk Scoring:** Each word gets a real-time risk score calculated from JLPT difficulty, review count, average mastery, and days since last review.
- **ML-Powered Memory Prediction:** Logistic Regression & Random Forest models trained on your interaction data predict memory retention probability.
- **Priority Queue:** Words with the highest forgetting risk surface first — no more wasting time on words you already know.

### 🎯 Adaptive Self-Testing
- **Dynamic Difficulty:** Question probability scales with your historical accuracy per word — weak words appear more often.
- **JLPT Level Estimation:** After enough answers, the system estimates your approximate JLPT vocabulary level (N5–N1).
- **Error Notebook:** Wrong answers are automatically collected into a focused review deck.

### 📊 Vocabulary Analytics
- **Word Frequency Analysis:** Paste any Japanese text and get tokenized word frequency, statistics, and a beautiful word cloud.
- **Learning Dashboard:** Track study days, review counts, accuracy trends, JLPT-level mastery breakdowns, and part-of-speech error distribution.
- **Data Health Check:** Built-in self-diagnosis detects missing fields, format errors, duplicates, and long-term low-mastery words.

### 📚 Vocabulary Management
- **Multi-source Import:** CSV/Excel import with auto-detection of column headers (`word/kana/meaning` or `单词/假名/释义`).
- **Built-in Word Banks:** Ships with curated decks: "Top 100 High-Frequency", "JLPT N4–N3 Extended", "News Reading High-Frequency".
- **Custom Decks:** Create, rename, and delete decks freely.

### 🔊 Pronunciation & Translation
- **TTS (Text-to-Speech):** Browser-native SpeechSynthesis API — works **offline**, no external API calls.
- **Translation Assistant:** Bundled offline-first translator app supporting Japanese ↔ Chinese with DeepSeek API fallback.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                       │
│  ┌──────────┬──────────┬──────────┬──────────────────┐ │
│  │  Smart   │ Adaptive │ Analytics│  Word Bank       │ │
│  │  Review  │  Test    │ & Charts │  Management      │ │
│  └────┬─────┴────┬─────┴────┬─────┴────────┬─────────┘ │
├───────┼──────────┼──────────┼──────────────┼───────────┤
│       ▼           ▼           ▼              ▼           │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SQLite Database                       │   │
│  │  words │ review_logs │ test_records │ error_book  │   │
│  └──────────────────────┬───────────────────────────┘   │
├─────────────────────────┼───────────────────────────────┤
│       ML Layer          │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │  Logistic Regression  │  Random Forest             │   │
│  │  Memory Retention     │  Forgetting Risk           │   │
│  │  Prediction           │  Classification            │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Supporting Modules                                      │
│  ┌────────────┐ ┌──────────┐ ┌──────────────────────┐  │
│  │ 翻译助手    │ │ Janome   │ │ WordCloud /          │  │
│  │ Translator │ │ Tokenizer│ │ Matplotlib           │  │
│  └────────────┘ └──────────┘ └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10+**
- Windows / macOS / Linux

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/myx1025377675-maker/japanese-vocab-memory-assistant.git
cd japanese-vocab-memory-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
streamlit run app.py
```

On Windows, you can also double-click `run_app.bat`.

> **Note:** On first launch, the app automatically creates `japanese_learning.db` with sample JLPT vocabulary and mock learning records. No internet connection is required for core features.

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application (156 KB)
├── requirements.txt            # Python dependencies
├── run_app.bat                 # Windows quick-launch script
├── README.md                   # This file
│
├── 翻译助手/                    # Translation Assistant sub-project
│   ├── translator.py           # Offline translator engine
│   ├── config.py               # Configuration management
│   ├── db.py                   # Database layer
│   └── prompts/
│       └── system_prompts.py   # LLM prompt templates
│
├── notes_cards/                # Pronunciation card generator
│   ├── generate_cards.ps1      # PowerShell card generation script
│   └── output/                 # Generated PNG cards
│       ├── 01_overview.png
│       ├── 02_final_vowels.png
│       ├── 03_initials.png
│       └── 04_sound_changes.png
│
├── 日语.txt                    # Sample Japanese text corpus
├── 演讲稿词汇表-v2.xlsx         # Vocabulary spreadsheet
└── 智能日语词汇记忆辅助系统_实验报告初版.md   # Project report (Chinese)
```

---

## 🔬 Machine Learning Approach

### Problem Formulation
Given a word and the learner's interaction history, predict whether the learner will remember the word at the next encounter. This is framed as a **binary classification** problem.

### Features
| Feature | Description |
|---------|-------------|
| `review_count` | Total number of times the word has been reviewed |
| `avg_mastery` | Average mastery score from past interactions |
| `days_since_last_review` | Days elapsed since the last review |
| `jlpt_level` | JLPT difficulty (N5=1 → N1=5) |
| `accuracy_history` | Historical accuracy on test questions |

### Models
| Model | Role | Notes |
|-------|------|-------|
| **Logistic Regression** | Baseline memory retention predictor | Interpretable, calibrated probabilities |
| **Random Forest** | Ensemble forgetting risk classifier | Captures non-linear patterns, robust to noise |

### Evaluation Metrics
- Accuracy, F1 Score, Precision, Recall (displayed in-app)
- Confusion matrix visualization
- Feature importance ranking (Random Forest)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **Database** | SQLite3 |
| **ML/AI** | [Scikit-learn](https://scikit-learn.org/), Logistic Regression, Random Forest |
| **Data Processing** | Pandas, NumPy |
| **NLP / Tokenization** | [Janome](https://mocobeta.github.io/janome/) (Japanese morphological analyzer) |
| **Visualization** | Matplotlib, [WordCloud](https://github.com/amueller/word_cloud) |
| **TTS** | Browser SpeechSynthesis API |
| **Translation** | Offline models + DeepSeek API fallback |
| **File Parsing** | OpenPyXL (Excel), csv (CSV), PyPDF2 (PDF) |

---

## 🎨 UI Design

The interface follows an **editorial magazine aesthetic** — warm paper tones, muted ink colors, and clean typography — designed for extended reading comfort during study sessions.

| Element | HEX |
|---------|-----|
| Paper Background | `#F5F2EB` |
| Panel | `#FAF8F5` |
| Primary Ink | `#2A2825` |
| Accent Red | `#8B2626` |
| Accent Blue | `#4F6575` |
| Accent Gold | `#9A7A3F` |

---

## 📝 License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

## 🙋‍♂️ Author

**Ma Yuexiang (马跃翔)**

- GitHub: [@myx1025377675-maker](https://github.com/myx1025377675-maker)
- Project Link: [japanese-vocab-memory-assistant](https://github.com/myx1025377675-maker/japanese-vocab-memory-assistant)

---

<p align="center">
  <sub>Built with ❤️ for Japanese learners everywhere. 日本語の勉強、頑張ってください！</sub>
</p>
