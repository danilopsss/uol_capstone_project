import os
import re

import ollama
import polars as pl
import textstat


class CapstoneLib:

    CLAUSE_DEPS = {
        "ROOT",
        "ccomp",
        "xcomp",
        "advcl",
        "relcl",
        "conj",
    }

    @classmethod
    def load_parquet(cls, parquet_path: str):
        if os.path.exists(parquet_path):
            return pl.read_parquet(parquet_path)

        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}"
        )

    @classmethod
    def paragraphs_extractor(cls, text):

        splitted = re.split(
            r"(\n\n|\t|\r|-*|\s*\n\s*)",
            text
        )

        return [
            re.sub(r"\s{2,}", " ", row)
            .lower()
            .strip()
            for row in splitted
            if "\n" not in row and row
        ]

    @classmethod
    def extract_clauses_from_doc(cls, doc):

        clauses = []

        for token in doc:

            if (
                token.pos_ == "VERB"
                and token.dep_ in cls.CLAUSE_DEPS
            ):
                span = doc[
                    token.left_edge.i:
                    token.right_edge.i + 1
                ]

                clauses.append(
                    span.text.lower().strip()
                )

        return clauses

    @classmethod
    def syntax_features(cls, doc):
        """
        Returns:
            (
                mdd,
                tree_depth,
                clause_density
            )
        """

        words = sum(
            1
            for token in doc
            if token.is_alpha
        )

        # Mean Dependency Distance
        deps = [
            abs(token.i - token.head.i)
            for token in doc
            if (
                token.head != token
                and token.dep_ != "punct"
            )
        ]

        mdd = (
            sum(deps) / len(deps)
            if deps
            else 0.0
        )

        # Tree Depth
        depths = [
            len(list(token.ancestors))
            for token in doc
            if token.head != token
        ]

        tree_depth = (
            max(depths)
            if depths
            else 0
        )

        # Clause Density
        markers = sum(
            1
            for token in doc
            if (
                token.pos_ == "VERB"
                and token.dep_ in cls.CLAUSE_DEPS
            )
        )

        clause_density = (
            markers / words
            if words
            else 0.0
        )

        return (
            mdd,
            tree_depth,
            clause_density,
        )

    @classmethod
    def add_readability_features(cls, df):

        return (
            df
            .with_columns(
                pl.col("sentences")
                .map_elements(
                    textstat.flesch_kincaid_grade,
                    return_dtype=pl.Float64,
                )
                .alias("fk_grade")
            )
            .with_columns(
                pl.col("sentences")
                .map_elements(
                    textstat.flesch_reading_ease,
                    return_dtype=pl.Float64,
                )
                .alias("f_reading_ease")
            )
            .with_columns(
                pl.col("sentences")
                .map_elements(
                    textstat.gunning_fog,
                    return_dtype=pl.Float64,
                )
                .alias("gunning_fog_idx")
            )
            .with_columns(
                pl.col("sentences")
                .map_elements(
                    textstat.dale_chall_readability_score,
                    return_dtype=pl.Float64,
                )
                .alias("dc_readability_score")
            )
            .with_columns(
                pl.col("sentences")
                .map_elements(
                    textstat.lix,
                    return_dtype=pl.Float64,
                )
                .alias("lix")
            )
            .with_columns(
                pl.col("sentences")
                .map_elements(
                    lambda sentence:
                    textstat.difficult_words(sentence)
                    / (textstat.lexicon_count(sentence) or 1),
                    return_dtype=pl.Float64,
                )
                .alias("word_density")
            )
        )

    @classmethod
    def add_syntax_features(
        cls,
        df,
        nlp,
        batch_size=256,
        n_process=4,
    ):

        unique_clauses = (
            df["sentences"]
            .unique()
            .to_list()
        )

        docs = list(
            nlp.pipe(
                unique_clauses,
                batch_size=batch_size,
                n_process=n_process,
            )
        )

        features = [
            cls.syntax_features(doc)
            for doc in docs
        ]

        syntax_df = pl.DataFrame({
            "sentences": unique_clauses,
            "mdd": [f[0] for f in features],
            "tree_depth": [f[1] for f in features],
            "clause_density": [f[2] for f in features],
        })

        return df.join(
            syntax_df,
            on="sentences",
            how="left",
        )

    @classmethod
    def add_complexity_features(
        cls,
        df,
        nlp,
        batch_size=256,
        n_process=4,
    ):

        df = cls.add_readability_features(df)

        df = cls.add_syntax_features(
            df=df,
            nlp=nlp,
            batch_size=batch_size,
            n_process=n_process,
        )

        return df

    @classmethod
    def llm_call(cls, clause_text):

        with open(
            "./prompts/clause_simplifyer.md",
            mode="r",
            encoding="utf-8",
        ) as f:
            system_prompt = f.read()

        response = ollama.chat(
            model="gemma4",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": clause_text,
                },
            ],
            options={
                "temperature": 0.0,
                "top_p": 0.9,
                "seed": 42,
                "num_predict": 2048,
            },
        )

        return response["message"]["content"]

    @classmethod
    def simplify_dataframe(
        cls,
        df,
        source_column="sentences",
        output_column="simplified_clause",
    ):

        return df.with_columns(
            pl.col(source_column)
            .map_elements(
                cls.llm_call,
                return_dtype=pl.String,
            )
            .alias(output_column)
        )