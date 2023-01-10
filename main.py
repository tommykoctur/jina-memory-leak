from docarray import Document, DocumentArray
from itertools import it
from jina import Flow


def doc_generator(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = f.readlines()
        for row in it.islice(data):
            d = Document(text=row)
            yield d


def index_data(file_path):

    with Flow.load_config("flow.yaml") as f:
        input_docs = doc_generator(filepath=file_path)
        f.post(
            on="/index",
            inputs=input_docs,
            request_size=1,
            show_progress=True,
        )
        print("Done")