from fastapi import FastAPI

import pydantic
import json

import joblib
import numpy as np

from sklearn.datasets import load_iris
from loguru import logger

import pandas as pd 
from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import aiplatform

model1 = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# The 94 embeddings are stored here
df = pd.read_csv('gs://your-bucket-name-inlaid-chassis-392615-unique/94_USCIS.csv')

app = FastAPI()


class QuestionSample(pydantic.BaseModel):
    query: str = pydantic.Field(default="What is the priority date for EB2 Mexico?", title="Query")

@app.get("/")
def hello_world():
    return {"message": "Hello World!"}





@app.post("/ann")
async def ann(sample: QuestionSample):
    logger.info("Query=" + sample.query)
    query_embedding = model1.get_embeddings([sample.query])[0].values
    scores = []
    for i, row in df.iterrows():
        scores.append(pd.Series(query_embedding).dot(pd.Series(row['embeddings'][1:-1].replace(" ", "").split(",")).astype(float)))
    titles = []    
    for index, (title, titledoc, score) in enumerate(
            sorted(zip(df[['Title']].values, df[['TitleDoc']].values, scores), key=lambda x: x[2], reverse=True)[:5]
    ):
        print(f"\t{index}: {titledoc}: {score}")    
        titles.append(str(title))
    print(titles)
    print(type(titles))
    return json.dumps({"prediction": titles})
