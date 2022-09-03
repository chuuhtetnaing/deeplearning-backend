from pdf2image import convert_from_bytes
import img2pdf
import pandas as pd
from nltk.tokenize import sent_tokenize

img_file = 'temp.jpg'
pdf_file = 'temp.pdf'


def pdf2images(pdf_byte, page_number):
    images = convert_from_bytes(pdf_byte)
    image = images[page_number-1]
    image.save(img_file)


def image2pdf():
    with open(pdf_file, 'wb') as f:
        f.write(img2pdf.convert(img_file))
    return pdf_file


def convertTableBlocksToHTML(page):
    raw_htmls = list()
    for table in page.tables:
        raw_htmls.append(table.html)
    return raw_htmls


def convertHTMLToDataFrame(raw_htmls):
    table_dfs = list()
    for raw_html in raw_htmls:
        table_df = pd.read_html(raw_html)[0]
        table_dfs.append(table_df)
    return table_dfs


def TableDataFrameToExcel(table_dfs):
    with pd.ExcelWriter('tables.xlsx') as writer:
        for i, table_df in enumerate(table_dfs):
            table_df.to_excel(writer, sheet_name=f'table-{i + 1}')
    return 'tables.xlsx'


def heuristics_paragraph_correction(paragraphs):
    corrected_paragraphs = []
    index = 0
    for i, p in enumerate(paragraphs):
        if i == 0:
            corrected_paragraphs.append(p)
            continue
        first_char_of_current = p[0]
        last_char_of_previous = corrected_paragraphs[-1][-1]

        if (first_char_of_current.islower()) & (last_char_of_previous != '.'):
            corrected_paragraphs[-1] = corrected_paragraphs[-1] + p
        else:
            corrected_paragraphs.append(p)
    return corrected_paragraphs


def paragraph_into_paragraph_dataframe(corrected_paragraphs):
    paragraph_df = pd.DataFrame(corrected_paragraphs, columns=['paragraph'])
    paragraph_df.reset_index(inplace=True)
    paragraph_df.rename(columns={'index': 'id'}, inplace=True)
    return paragraph_df


def paragraph_into_sentence_dataframe(paragraph_df):
    sentence_dfs = [list() for i in range(paragraph_df.shape[0])]
    for i, paragraph in paragraph_df.iterrows():
        paragraph = paragraph[1]
        sentences = sent_tokenize(paragraph)
        sentence_dfs[i] = pd.DataFrame(sentences, columns=['sentence'])
    return sentence_dfs


def export_par_dataframe_into_excel(paragraph_df):
    paragraph_df.to_excel('paragraph.xlsx')
    return 'paragraph.xlsx'


def export_sent_dataframe_into_excel(sentence_dfs):
    with pd.ExcelWriter('sentence.xlsx') as writer:
        for i, sentence_df in enumerate(sentence_dfs):
            sentence_df.to_excel(writer, sheet_name=f'paragraph-{i+1}')
    return 'sentence.xlsx'
