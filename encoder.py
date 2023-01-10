#modified from https://github.com/jina-ai/executor-text-transformers-torch-encoder/blob/main/transform_encoder.py
__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from docarray import DocumentArray, Document
from jina import Executor, requests


class TransformerTorchEncoder(Executor):
    """The TransformerTorchEncoder encodes sentences into embeddings using transformers models."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'sentence-transformers/all-mpnet-base-v2',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        embedding_fn_name: str = '__call__',
        device: str = 'cpu',
        traversal_paths: str = '@r',
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        The transformer torch encoder encodes sentences into embeddings.

        :param pretrained_model_name_or_path: Name of the pretrained model or path to the
            model
        :param base_tokenizer_model: Base tokenizer model
        :param pooling_strategy: The pooling strategy to be used. The allowed values are
            ``'mean'``, ``'min'``, ``'max'`` and ``'cls'``.
        :param layer_index: Index of the layer which contains the embeddings
        :param max_length: Max length argument for the tokenizer, used for truncation. By
            default the max length supported by the model will be used.
        :param embedding_fn_name: Function to call on the model in order to get output
        :param device: Torch device to put the model on (e.g. 'cpu', 'cuda', 'cuda:1')
        :param traversal_paths: Used in the encode method an define traversal on the
             received `DocumentArray`
        :param batch_size: Defines the batch size for inference on the loaded
            PyTorch model.
        """
        super().__init__(*args, **kwargs)

        self.traversal_paths = traversal_paths
        self.batch_size = batch_size

        base_tokenizer_model = base_tokenizer_model or pretrained_model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length

        self.device = device
        self.embedding_fn_name = embedding_fn_name

        self.model = SentenceTransformer(pretrained_model_name_or_path, device=device)
        self.model.to(device).eval()

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict={}, **kwargs):
        """
        Encode text data into a ndarray of `D` as dimension, and fill the embedding of
        each Document.

        :param docs: DocumentArray containing text
        :param parameters: dictionary to define the `traversal_paths` and the
            `batch_size`. For example,
            `parameters={'traversal_paths': 'r', 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """

        for doc in docs[self.traversal_paths]:

            with torch.inference_mode():
                embedding = self.model.encode(doc.text, output_value=None, show_progress_bar=False)

                sentence_embedding = embedding["sentence_embedding"].cpu().numpy()
                token_embeddings = embedding["token_embeddings"].cpu().numpy()

                # trim cls and sep tokens
                token_embeddings = token_embeddings[1:-1]

                if self.normalize_token_embeddings:
                    token_embeddings = (
                            token_embeddings / np.linalg.norm(token_embeddings, axis=1)[:, None]
                    )

                if self.use_fp16:
                    token_embeddings = token_embeddings.astype(np.half)

            doc.tensor = token_embeddings
            doc.embedding = sentence_embedding
