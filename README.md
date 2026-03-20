# TylerBot 🤖

A multi-model Discord chatbot that lets users interact with different AI personalities and architectures through slash commands.

TylerBot is designed as an experimental playground for comparing:

* Custom-trained LLMs
* Classic transformer models (GPT-2)
* Modern instruction-tuned models

---

## 🚀 Features

### 🔀 Multi-Model System

Users can dynamically choose between **three distinct AI behaviors**:

| Command      | Model Type              | Behavior                                    |
| ------------ | ----------------------- | ------------------------------------------- |
| `/legacy`    | Custom Local LLM        | Stylized, personality-heavy, less realistic |
| `/realistic` | GPT-2 (HuggingFace)     | More natural language, basic coherence      |
| `/latest`    | Instruction-tuned model | Chaotic, energetic, modern responses        |

---

### 🧠 Model Breakdown

#### 1. `/legacy` — Local Tyler Model

* Built using your custom `GPTLanguageModel`
* Uses `RegexTokenizer` (minBPE)
* Trained checkpoint loaded locally
* Strong formatting system:

  * `<|startoftext|>`
  * `<|separator|>`
* Output cleaned into:

  ```
  You: ...
  Tyler: ...
  ```

**Use case:** Testing your own architecture + stylized outputs

---

#### 2. `/realistic` — GPT-2

* Uses `GPT2LMHeadModel` from HuggingFace
* Prompt format:

  ```
  User: <input>
  You:
  ```
* Output is **post-processed to ONLY show the model reply**
* Tuned for:

  * Better coherence than local model
  * Reduced repetition (via penalties)

**Use case:** Baseline realism comparison

---

#### 3. `/latest` — Modern Instruction Model

* Model: `Qwen/Qwen2.5-0.5B-Instruct`
* Uses chat-template formatting (role आधारित prompting)
* System prompt defines Tyler personality:

  * Hyperactive
  * Chaotic
  * Over-the-top responses
  * Injects randomness (e.g., “sushi” keyword)

**Use case:** Personality-driven + modern LLM behavior

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo>
cd TylerBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required libraries:**

* discord.py
* torch
* transformers

---

### 3. Configure Paths

Update these paths in code:

```python
# Tokenizer
model_file="PATH_TO_TOKENIZER.model"

# Checkpoint
checkpoint_path="PATH_TO_CHECKPOINT.pth"
```

---

### 4. Add Discord Token

Replace:

```python
bot.run("REDACTED")
```

with your actual bot token.

---

## ▶️ Running the Bot

```bash
python your_script.py
```

Expected output:

```
✅ Local Tyler model loaded
✅ GPT-2 model loaded
✅ Modern model loaded: Qwen/Qwen2.5-0.5B-Instruct
✅ Slash commands synced
✅ BotName is online
```

---

## 💬 Usage

Inside Discord:

### Core Commands

```
/legacy message: Hello
/realistic message: Hello
/latest message: Hello
```

---

### Help Command

```
/help
```

Displays model differences and usage guide.

---

## ⚙️ Architecture Overview

```
User Input (Discord Slash Command)
        ↓
Command Router (/legacy | /realistic | /latest)
        ↓
Model Wrapper (Local | GPT2 | Qwen)
        ↓
Prompt Formatting
        ↓
Text Generation (Torch / HF)
        ↓
Output Cleaning
        ↓
Discord Response
```

---

## 🧪 Key Design Decisions

### 1. Model Wrappers

Each model is encapsulated:

```python
class LocalTylerModel
class GPT2TylerModel
class ModernTylerModel
```

This allows:

* Clean abstraction
* Easy swapping
* Independent tuning

---

### 2. Output Cleaning Pipelines

Each model has **custom post-processing**:

| Model  | Cleaning Strategy                 |
| ------ | --------------------------------- |
| Local  | Token stripping + role formatting |
| GPT-2  | Extract only "You:" response      |
| Modern | Remove assistant prefixes         |

---

### 3. Prompt Engineering

* Local → structured tokens
* GPT-2 → conversational imitation
* Modern → system + role-based prompting

---

## ⚠️ Known Limitations

### GPT-2

* Weak long-term coherence
* May hallucinate dialogue structure
* Occasionally repeats "You:" lines (partially mitigated)

---

### Local Model

* Highly dependent on training quality
* Can produce noise or malformed structure

---

### Modern Model

* Personality heavily influenced by system prompt
* Can become chaotic by design

---

## 🔮 Future Improvements

* Add model switching mid-conversation
* Add conversation memory (context window management)
* Streaming responses
* UI buttons instead of slash commands
* Fine-tuned GPT-2 replacement
* Quantized model loading for performance

---

## 👤 Credits

* Developed by **Tanish (DiamondMiner / HaloVision)**
* Built using:

  * PyTorch
  * HuggingFace Transformers
  * Discord API

---

## 🧠 Philosophy

TylerBot is not just a chatbot.

It is a **controlled experiment in alignment**:

* What makes text feel human?
* How does structure influence perception?
* Where does personality emerge from?

---

If you want, next step I’d strongly recommend:
→ Add a **live model switcher per user session** (not per command). That’s where this becomes seriously powerful.
