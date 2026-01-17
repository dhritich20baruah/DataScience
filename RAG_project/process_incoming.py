import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post('http://localhost:11434/api/embed', json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

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

prompt = f'''I am teaching electronics using udemy. Here are the video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json()}
-------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
'''

for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])