# modified from https://github.com/alexcg1/executor-spacy-sentencizer/blob/main/executor.py
from jina import Executor, requests
from typing import Any, Dict, Optional, Sequence
from docarray import Document, DocumentArray
from spacy.lang.en import English
import re


class SpacySentencizer(Executor):
    """
    Splits text at doc-level into sentences using spaCy's sentencizer and stores as doc.chunks.
    """

    def __init__(self, min_sent_len=20, traversal_paths: str = "@r", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sent_len = min_sent_len
        self.traversal_paths = traversal_paths
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

    @requests(on="/index")
    def segment(self, docs: DocumentArray, parameters: Dict[str, Any], **kwargs):
        traversal_paths = parameters.get("traversal_paths", self.traversal_paths)
        # First do some cleanup of unwanted linebreaks
        substitutions = [
            # Convert deliberate paragraph breaks into sentence-ends so they dont get caught by next entry in the list
            {"old": "\n\n.*", "new": ". "},
            {"old": "\r\r.*", "new": ". "},
            # Remove incidental linebreaks caused by conversion
            {"old": "\n", "new": " "},
            {"old": "\r", "new": " "},
            {"old": "\\s+", "new": " "},  # collapse white-spaces into one space
        ]

        for doc in docs[traversal_paths]:
            if doc.text:
                for sub in substitutions:
                    doc.text = re.sub(sub["old"], sub["new"], doc.text)

                # Sentencize
                text = self.nlp(doc.text)

                for sent in text.sents:
                    if len(str(sent)) >= self.min_sent_len:
                        doc.chunks.append(Document(text=str(sent)))