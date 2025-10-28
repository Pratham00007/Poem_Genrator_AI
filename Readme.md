# Text Generator AI 🤖✍️

An LSTM-based neural network that learns writing patterns and generates new text in a similar style. Train it on any text corpus - Shakespeare, your WhatsApp chats, books, or any text you want!

## 📋 Project Overview

This project uses a **Long Short-Term Memory (LSTM)** recurrent neural network to:
- Learn character-level patterns from text data
- Predict the next character based on previous 40 characters
- Generate new text that mimics the training data's style

## 🎯 Features

- **Character-level text generation** using LSTM
- **Temperature-based creativity control** (0.2 = conservative, 1.0 = creative)
- **Customizable training** on any text corpus
- **Pre-trained model** saving and loading
- **One-hot encoding** for character representation

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy
```

### Files in Project

- **`main.ipynb`** - Training the model (LSTM architecture, data preprocessing)
- **`2.ipynb`** - Generating text using the trained model
- **`textgenerator.h5`** - Pre-trained model weights
- **`Readme.md`** - Project documentation

## 📚 How to Use

### Option 1: Using Shakespeare Text (Default)

The model downloads Shakespeare's text automatically:

```python
filepath = tf.keras.utils.get_file('shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

### Option 2: Using Your Own WhatsApp Chat 💬

**Step 1: Export WhatsApp Chat**
1. Open WhatsApp conversation
2. Tap on ⋮ (three dots) → More → Export chat
3. Choose "Without Media"
4. Save the `.txt` file

**Step 2: Clean the Data**

Create a new cell in `main.ipynb` and add:

```python
# Read WhatsApp chat export
with open('your_whatsapp_chat.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Clean the text (remove timestamps, names, etc.)
import re

# Remove timestamps like [12/25/23, 10:30:45 AM]
text = re.sub(r'\[\d+/\d+/\d+,\s\d+:\d+:\d+\s[AP]M\]', '', raw_text)

# Remove usernames (e.g., "John: ")
text = re.sub(r'^[^:]+:\s', '', text, flags=re.MULTILINE)

# Remove media messages
text = re.sub(r'<Media omitted>', '', text)
text = re.sub(r'image omitted', '', text)

# Convert to lowercase
text = text.lower()

# Remove extra whitespace
text = ' '.join(text.split())

print(f"Text length: {len(text)} characters")
```

**Step 3: Replace the filepath section** in `main.ipynb`:

Replace this:
```python
filepath = tf.keras.utils.get_file('shakespeare.txt',...)
text = open(filepath,'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]
```

With this:
```python
# Use your cleaned WhatsApp text
# (already loaded and cleaned in previous cell)
# Optionally slice it if it's too long
if len(text) > 100000:
    text = text[:100000]  # Use first 100k characters
```

**Step 4: Train the model** - Run all cells in `main.ipynb`

**Step 5: Generate text** - Run cells in `2.ipynb` to see your AI mimic your chat style!

## 🏗️ Model Architecture

```
Sequential Model:
├── LSTM Layer (128 neurons)
│   └── Input: (40 characters, one-hot encoded)
├── Dense Layer (# of unique characters)
└── Softmax Activation (probability distribution)
```

**Hyperparameters:**
- Sequence Length: 40 characters
- Step Size: 3 (sliding window)
- Batch Size: 256
- Epochs: 4
- Optimizer: RMSprop (learning rate: 0.01)
- Loss: Categorical Crossentropy

## 🎨 Temperature Control

Temperature controls creativity vs. accuracy:

| Temperature | Behavior | Best For |
|------------|----------|----------|
| **0.2** | Conservative, safe predictions | Coherent, grammatical text |
| **0.4-0.6** | Balanced creativity | Good general-purpose setting |
| **0.8-1.0** | Highly creative, risky | Experimental, diverse output |

## 📊 Example Usage

```python
# Generate 300 characters with low temperature (safe)
print(generate_text(300, temperature=0.2))

# Generate 300 characters with high temperature (creative)
print(generate_text(300, temperature=1.0))
```

## 🔬 How It Works

1. **Data Preprocessing**
   - Text is converted to lowercase
   - Unique characters are extracted
   - Character ↔ Index mappings are created

2. **Sequence Creation**
   - Text is split into sequences of 40 characters
   - Each sequence predicts the 41st character
   - Sliding window with step size of 3

3. **One-Hot Encoding**
   - Each character is represented as a binary vector
   - Input shape: `(num_sequences, 40, num_unique_chars)`
   - Output shape: `(num_sequences, num_unique_chars)`

4. **Training**
   - LSTM learns patterns from sequences
   - Model predicts probability distribution for next character

5. **Generation**
   - Start with random 40-character seed
   - Predict next character, append it
   - Slide window, repeat

## 📝 Tips for Better Results

- **More training data** = Better results (aim for 100k+ characters)
- **Clean your data** - Remove irrelevant content, timestamps, etc.
- **Train longer** - Increase epochs (4 → 10+) for better learning
- **Adjust temperature** - Experiment with different values
- **Use consistent text style** - Similar writing style in training data helps

## 🛠️ Troubleshooting

**Problem:** Generated text is gibberish
- **Solution:** Train longer, use more data, or lower temperature

**Problem:** Text is too repetitive
- **Solution:** Increase temperature or train on more diverse data

**Problem:** Model takes too long to train
- **Solution:** Use smaller text subset or reduce epochs

## 🎓 Learning Resources

- **LSTM Networks**: Understanding recurrent neural networks
- **One-Hot Encoding**: Representing categorical data
- **Temperature Sampling**: Controlling randomness in predictions

## 📜 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

## 🎉 Fun Ideas

- Train on your favorite book series
- Create a bot that talks like you (WhatsApp data)
- Generate poetry in specific styles
- Create song lyrics generator
- Build a code completion tool

---

**Made with ❤️ using TensorFlow & LSTM**

*Remember: The more data you feed it, the smarter it gets!* 🧠