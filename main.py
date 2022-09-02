from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
from nerPrePostProcessing import preprocess_sentence, postprocess_predictions

ner_model = load('model/ner_model_bert_base_uncased.joblib')
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


class NEROut(BaseModel):
    original: str
    entity: str
    word: str
    start: int
    end: int


class NERIn(BaseModel):
    text: str


@app.post("/ner", response_model=List[List[NEROut]])
async def ner(body: NERIn):
    # time.sleep(2)
    print("***********")
    text = body.text
    sentences = preprocess_sentence(text)
    print('sentences', sentences)
    predictions, _ = ner_model.predict(sentences)
    print('predictions', predictions)
    result = postprocess_predictions(predictions, sentences)
    print("***********")

    return result
