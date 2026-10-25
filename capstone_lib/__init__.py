import os
import re
import polars as pl
import spacy
import ollama


class CapstoneLib:
    
    CLAUSE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "conj"}

    @classmethod
    def load_parquet(cls, parquet_name: str):
        if os.path.exists(f"../parquet_files/{parquet_name}"):
            return polars.read_parquet(parquet_name)

    @classmethod
    def paragraphs_extractor(cls, text):
        splitted = re.split(r"(\n\n|\t|\r|-*|\s*\n\s*)", text)
        return [
            re.sub(r"\s{2,}", " ", row).lower().strip()
            for row in splitted if "\n" not in row and row
        ]

    @classmethod
    def extract_clauses_from_doc(cls, doc):
        clauses = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in CLAUSE_DEPS:
                span = doc[token.left_edge.i : token.right_edge.i + 1]
                clauses.append(span.text.lower())
        return clauses
    
    @classmethod
    def syntax_features(cls, doc):
        words = sum(1 for t in doc if t.is_alpha)
        deps = [abs(t.i - t.head.i) for t in doc if t.head != t and t.dep_ != "punct"]
        depths = [len(list(t.ancestors)) for t in doc if t.head != t]
        markers = sum(1 for t in doc if t.pos_ == "VERB" and t.dep_ in cls.CLAUSE_DEPS)
        return (
            sum(deps) / len(deps) if deps else 0.0,
            max(depths) if depths else 0,
            markers / words if words else 0.0,
        )


    @classmethod
    def llm_call(cls, user_prompt):
        with open("./prompts/clause_simplifyer.md", mode="r") as f:
            system_prompt = f.read()
        
        response = ollama.chat(
            model="gemma4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cuad_legal[5,2]}
            ],
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": 42,
                "num_predict": 2048
            }
        )
        
        return response["message"]
