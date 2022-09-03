from fastapi import FastAPI, UploadFile
from typing import List
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
from pdfProcessing import pdf2images, image2pdf, \
    convertTableBlocksToHTML, convertHTMLToDataFrame, \
    TableDataFrameToExcel, heuristics_paragraph_correction, \
    paragraph_into_paragraph_dataframe, paragraph_into_sentence_dataframe, \
    export_par_dataframe_into_excel, export_sent_dataframe_into_excel
from starlette.responses import FileResponse
from nerPrePostProcessing import preprocess_sentence, postprocess_predictions
from deepdoctection.analyzer import get_dd_analyzer

ner_model = load('model/ner_model_bert_base_uncased.joblib')
app = FastAPI()
analyzer = get_dd_analyzer()

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
    print("***********")
    text = body.text
    sentences = preprocess_sentence(text)
    print('sentences', sentences)
    predictions, _ = ner_model.predict(sentences)
    print('predictions', predictions)
    result = postprocess_predictions(predictions, sentences)
    print("***********")

    return result


@app.post("/uploadfiles/{req_type}/{page_number}")
async def create_upload_files(files: List[UploadFile], req_type: str, page_number: int):
    pdf_byte = await files[0].read()
    pdf2images(pdf_byte, page_number)
    pdf_file = image2pdf()
    df = analyzer.analyze(path=pdf_file)

    doc = iter(df)
    page = next(doc)

    if (req_type == 'paragraph') | (req_type == 'sentence'):
        raw_text = page.get_text()
        paragraphs = raw_text.split('\n')
        paragraphs = [paragraph.strip() for paragraph in paragraphs if len(paragraph)]
        corrected_paragraphs = heuristics_paragraph_correction(paragraphs)
        paragraph_df = paragraph_into_paragraph_dataframe(corrected_paragraphs)
        sentence_dfs = paragraph_into_sentence_dataframe(paragraph_df)
        if req_type == 'paragraph':
            excel_file = export_par_dataframe_into_excel(paragraph_df)
        else:
            excel_file = export_sent_dataframe_into_excel(sentence_dfs)
    else:
        raw_htmls = convertTableBlocksToHTML(page)
        table_dfs = convertHTMLToDataFrame(raw_htmls)
        excel_file = TableDataFrameToExcel(table_dfs)

    return FileResponse(f"./{excel_file}")


