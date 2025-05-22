import openai
import os
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv 

# Load environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_relevant_context(question, contexts, top_k=1):
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, context_embeddings)[0]
    top_results = scores.topk(k=top_k)
    return [contexts[idx] for idx in top_results.indices]

def ask_llm_with_context(paragraphs, question):
    relevant_paras = get_most_relevant_context(question, paragraphs)
    prompt = f"""Answer the question based on the following context:\n\n{relevant_paras[0]}\n\nQuestion: {question}\nAnswer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return relevant_paras[0], response['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    paragraphs = [
        "Augmentin is a combination antibiotic containing amoxicillin and clavulanate. It is used to treat various bacterial infections, including respiratory tract infections, urinary tract infections, and skin infections.",
        "Penicillin is one of the earliest discovered and widely used antibiotic agents. It primarily treats gram-positive bacterial infections.",
        "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for relieving pain, fever, and inflammation."
    ]

    question = input("Enter your question: ")
    context, answer = ask_llm_with_context(paragraphs, question)

    print(f"\nRelevant Context:\n{context}")
    print(f"\nAnswer: {answer}")
