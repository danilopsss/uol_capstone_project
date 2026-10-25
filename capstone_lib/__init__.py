import functools
import os
import re

import ollama
import polars
import spacy
import textstat
from spacy.language import Language
from spacy.tokens import Doc


class CapstoneLib:

    CLAUSE_DEPS = {
        "ROOT",
        "ccomp",
        "xcomp",
        "advcl",
        "relcl",
        "conj",
    }

    CLAUSE_PIPE = "capstone_clause_extractor"
    CLAUSE_EXT = "clauses"
    PARQUET_PATH = "parquet_files/0002_cuad_dataset.parquet"

    @classmethod
    def load_parquet(cls, parquet_path: str):
        if os.path.exists(parquet_path):
            return polars.read_parquet(parquet_path)
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}"
        )

    @classmethod
    def apply_metrics(cls, df: polars.DataFrame):
        cuad_metrics =  df.with_columns(
            polars.col("sentences")
            .map_elements(
                lambda x: textstat.flesch_reading_ease(x),
                return_dtype=polars.Float64,
            )
            .alias("fre"),
        
            polars.col("sentences")
            .map_elements(
                lambda x: textstat.flesch_kincaid_grade(x),
                return_dtype=polars.Float64,
            )
            .alias("fkgl"),
        
            polars.col("sentences")
            .map_elements(
                lambda x: textstat.gunning_fog(x),
                return_dtype=polars.Float64,
            )
            .alias("fog"),
        
            polars.col("sentences")
            .map_elements(
                lambda x: textstat.dale_chall_readability_score(x),
                return_dtype=polars.Float64,
            )
            .alias("dale_chall"),
        
            polars.col("sentences")
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
