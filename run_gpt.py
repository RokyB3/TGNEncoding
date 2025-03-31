from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Ex prompt
graph_text = "The graph at time t=0 contains edges (A-B), (B-C)."
question = "What was the graph structure at t=0?"

response = client.chat.completions.create(
    model="gpt-4-turbo",  # or "gpt-4" if using the older GPT-4
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers questions about temporal graphs."},
        {"role": "user", "content": f"{graph_text}\n\nQuestion: {question}"}
    ],
    temperature=0
)

print(response.choices[0].message.content)
