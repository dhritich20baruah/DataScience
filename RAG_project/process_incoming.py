import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def create_embedding(text_list):
    r = requests.post('http://localhost:11434/api/embed', json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

model_name = "gemini-3-flash-preview"

def inference(prompt):
    r = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    response = r.text
    print(response)
    return response

df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
max_index = similarities.argsort()[::-1][0:3]
# print(max_index)
new_df = df.loc[max_index]
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching electronics in udemy. Here are the video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
-------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way, where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)
print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])