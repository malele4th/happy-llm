#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from config import EMBEDDING_MODEL

_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        if magnitude == 0:
            return 0.0
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        if self.is_api:
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        raise NotImplementedError
