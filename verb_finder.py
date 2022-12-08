from __future__ import unicode_literals, print_function
from functools import partial
import re
import pdfplumber
import spacy
from openpyxl.workbook import Workbook
from openpyxl.styles import Font
from table_handler import get_bounding_boxes, table_filter
import time

FILTER_PATTERNS = [
    r"Figure\s[0-9]+\s[–-]\s.+",  # Figure title filter
    r"Table\s[0-9]+\s[–-]\s.+",  # Table title filter
    r"[0-9]+\.[0-9]+\.[0-9]+\s{6}.+",  # subsection filter
    r"[0-9]+\.[0-9]+\s{6}.+",  # sub sub section filter
    r"HiQPdf\sEvaluation\s\d\d\/\d\d\/\d\d\d\d"  # special filter for online converted pdf
]

SPACY_MODEL = 'en_core_web_sm'


def text_cleaner(txt, patterns):
    return re.sub('|'.join(patterns), "", txt)


def setup_nlp_pipeline(spacy_model, use_sentencizer=True):
    if not spacy.util.is_package(spacy_model):
        spacy.cli.download(spacy_model)
    nlp_pipeline = spacy.load(spacy_model)
    if use_sentencizer:
        nlp_pipeline.add_pipe('sentencizer')
    return nlp_pipeline


def get_verbs(sentence, lemmatize=True, detect_aux=True):
    verb_types = ["VERB", "AUX"] if detect_aux else ["VERB"]
    if lemmatize:
        return [word.lemma_ for word in sentence if word.pos_ in verb_types]
    else:
        return [word for word in sentence if word.pos_ in verb_types]


def process_document(document, filter_single_word=False):
    if filter_single_word:
        return [(sentence.text.strip().replace('\n', ''), get_verbs(sentence))
                for sentence in list(document.sents)
                if len(sentence.text.strip().replace('\n', '').split()) > 1]
    else:
        return [(sentence.text.strip().replace('\n', ''), get_verbs(sentence))
                for sentence in list(document.sents)]


def save_xlsx(table):
    wb = Workbook()
    ws = wb.active  # grab the active worksheet
    ws['A1'] = "Sentences"
    ws['A1'].font = Font(bold=True)
    ws['B1'] = "Verb-entities"
    ws['B1'].font = Font(bold=True)
    for i, (sentence, verbs) in enumerate(table, 2):
        ws[f"A{i}"] = sentence
        ws[f"B{i}"] = ", ".join(verbs)
    wb.save("result.xlsx")


def driver():
    # open pdf
    pdf = pdfplumber.open("test_pdf.pdf")

    # keep nlp pipeline ready
    nlp = setup_nlp_pipeline(SPACY_MODEL)

    # loop pages
    clean_text = ""
    for p in pdf.pages[17:23]:  # 6.3 starts at page no 18 and ends at 24. Page count starts from 1
        # find table locations if any
        bboxes = get_bounding_boxes(p)
        page_specific_table_filter = partial(table_filter, bboxes)
        # find the text not lying in the table region
        text = p.filter(page_specific_table_filter).extract_text()
        # clean text and append
        clean_text = '\n\n'.join([clean_text, text_cleaner(text, FILTER_PATTERNS)])

    # run nlp pipeline
    doc = nlp(clean_text)
    table = process_document(doc)
    save_xlsx(table)


if __name__ == "__main__":
    start = time.time()
    driver()
    print(f"Time elapsed: {time.time()-start} seconds")

