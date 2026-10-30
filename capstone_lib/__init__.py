import functools
import os
import re

import ollama
import polars
import spacy
import textstat
import requests

from functools import lru_cache
from pathlib import Path
from spacy.language import Language
from spacy.tokens import Doc


class CapstoneLib:

    PROJ_ROOT = "/workspace"
    
    CLAUSE_DEPS = {"ROOT", "ccomp" ,"xcomp" ,"advcl" ,"relcl" ,"conj"}
    CLAUSE_PIPE = "capstone_clause_extractor"
    CLAUSE_EXT = "clauses"

    LLM_TEMP = 0.7
    OLLAMA_URL = "http://localhost:11434/api/chat"

    @classmethod
    def load_parquet(cls, parquet_path: str):
        if os.path.exists(parquet_path):
            return polars.read_parquet(parquet_path)
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}"
        )

    @classmethod
    def apply_metrics(cls, df: polars.DataFrame, column: str = "sentences"):
        cuad_metrics =  df.with_columns(
            polars.col(column)
            .map_elements(
                lambda x: textstat.flesch_reading_ease(x),
                return_dtype=polars.Float64,
            )
            .alias("fre"),
        
            polars.col(column)
            .map_elements(
                lambda x: textstat.flesch_kincaid_grade(x),
                return_dtype=polars.Float64,
            )
            .alias("fkgl"),
        
            polars.col(column)
            .map_elements(
                lambda x: textstat.gunning_fog(x),
                return_dtype=polars.Float64,
            )
            .alias("fog"),
        
            polars.col(column)
            .map_elements(
                lambda x: textstat.dale_chall_readability_score(x),
                return_dtype=polars.Float64,
            )
            .alias("dale_chall"),
        
            polars.col(column)
            .map_elements(
                lambda x: textstat.lix(x),
                return_dtype=polars.Float64,
            )
            .alias("lix"),
        )
        return (
            cuad_metrics
                .with_columns(
                (
                    0.20 * polars.col("fre")
                    + 0.20 * polars.col("fkgl")
                    + 0.20 * polars.col("fog")
                    + 0.20 * polars.col("dale_chall")
                    + 0.20 * polars.col("lix")
                ).alias("complexity_score")
            )
        )

    @classmethod
    @lru_cache
    def get_system_prompt(cls, prompt_path: str):
        with open(prompt_path, mode="r") as prompt:
            return prompt.read()
        raise FileNotFoundError("Prompt not found")

    @classmethod
    def llm_call(cls, message: str, prompt_path: str, model: str = "gemma4"):
        system_prompt = cls.get_system_prompt(prompt_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        resp = requests.post(cls.OLLAMA_URL, json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": cls.LLM_TEMP},
        })
        if resp.ok:
            resp_json = resp.json()
            return resp_json.get("message")

    @classmethod
    def llm_call_with_progress_bar(cls, tqdm_bar, sentence: str, prompt_path: str):
        result = cls.llm_call(sentence, prompt_path)
        tqdm_bar.update(1)
        return result.get("content")