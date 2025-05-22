# LLM Q&A with Context

This project allows you to ask questions based on a given context using an LLM (like GPT-3.5) and retrieves the most relevant paragraph using semantic similarity.

## Features

- Uses OpenAI's GPT models
- Retrieves the most relevant context using Sentence Transformers
- Easy to run and modify

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd QNA_Context
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Rename `.env.example` to `.env` and add your OpenAI API key:

```bash
mv .env.example .env
```

### 5. Run the script

```bash
python qna_with_context.py
```

## Sample Input

```
Enter your question: What is Augmentin used for?
```

## Sample Output

```
Relevant Context:
Augmentin is a combination antibiotic...

Answer:
Augmentin is used to treat various bacterial infections...
```
