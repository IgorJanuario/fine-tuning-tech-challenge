import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)

TRN_JSON_PATH = "data/trn.json"
OUTPUT_DIR = "outputs/flan_t5_base_tech_challenge"
BASE_MODEL = "google/flan-t5-base"

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 256

MAX_SAMPLES = None

def load_trn(trn_path: str) -> pd.DataFrame:
    df = pd.read_json(trn_path, lines=True)
    keep = [c for c in df.columns if c in ("title", "content")]
    df = df[keep].dropna(subset=["title", "content"])
    df["title"] = df["title"].astype(str).str.strip()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[(df["title"].str.len() > 2) & (df["content"].str.len() > 10)]
    if MAX_SAMPLES is not None:
        df = df.sample(n=MAX_SAMPLES, random_state=SEED)
    df = df.reset_index(drop=True)
    return df

QUESTION_TEMPLATES = [
    "What is this product: {title}?",
    "Describe the product: {title}.",
    "List key details about {title}.",
    "Tell me about: {title}",
    "What are the features of {title}?",
    "Give me information about {title}",
]

def quick_check(df):
    print("Verificação rápida \n")
    print("Total bruto:", len(df))
    print("Nulos por coluna:\n", df.isnull().sum())
    
    df["len_title"] = df["title"].str.len()
    df["len_content"] = df["content"].str.len()
    print("Tamanho médio do título:", df["len_title"].mean())
    print("Tamanho médio da descrição:", df["len_content"].mean())

    print("Amostra de dados carregada com sucesso!")

def build_supervised_pairs(df: pd.DataFrame) -> pd.DataFrame:
    inputs, targets = [], []
    for _, row in df.iterrows():
        title = row["title"]
        content = row["content"]

        templates = random.sample(QUESTION_TEMPLATES, k=3)
        for tpl in templates:
            question = tpl.format(title=title)
            inp = f"Answer the user question using the product description.\nQuestion: {question}\n"
            tgt = content
            inputs.append(inp)
            targets.append(tgt)

    return pd.DataFrame({"input_text": inputs, "target_text": targets})

def to_hf_dataset(df_pairs: pd.DataFrame) -> DatasetDict:
    ds = Dataset.from_pandas(df_pairs)
    ds = ds.train_test_split(test_size=0.1, seed=SEED)
    test_valid = ds["test"].train_test_split(test_size=0.5, seed=SEED)
    return DatasetDict(train=ds["train"], validation=test_valid["train"], test=test_valid["test"])

def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_rouge(eval_preds, tokenizer):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            score = scorer.score(pred, label)
            scores.append(score['rougeL'].fmeasure)
        return {"rougeL": sum(scores) / len(scores)}
    except ImportError:
        print("Aviso: rouge_score não disponível, retornando métrica dummy")
        return {"rougeL": 0.0}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Carregando trn.json ...")
    df_raw = load_trn(TRN_JSON_PATH)
    quick_check(df_raw)
    print(f"Registros válidos: {len(df_raw)}")

    print("Gerando pares (pergunta sobre título -> descrição)...")
    df_pairs = build_supervised_pairs(df_raw)
    print(f"Total de pares gerados: {len(df_pairs)}")

    dsets = to_hf_dataset(df_pairs)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    tok_train = dsets["train"].map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dsets["train"].column_names)
    tok_val   = dsets["validation"].map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dsets["validation"].column_names)
    tok_test  = dsets["test"].map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dsets["test"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=2,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        include_inputs_for_metrics=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_rouge(p, tokenizer),
    )

    print("Treinando...")
    trainer.train()

    print("Avaliando (validação/teste)...")
    val_metrics = trainer.evaluate(eval_dataset=tok_val, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=tok_test, metric_key_prefix="test")
    print("VALIDAÇÃO:", val_metrics)
    print("TESTE:", test_metrics)

    print("Salvando modelo/tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    demo_titles = df_raw["title"].sample(3, random_state=SEED).tolist()
    questions = [f"What is this product: {t}?" for t in demo_titles]
    print("\n=== DEMONSTRAÇÃO ===")
    for q in questions:
        prompt = f"Answer the user question using the product description.\nQuestion: {q}\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
        outputs = model.generate(**inputs, max_new_tokens=192)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"P: {q}\nR: {answer}\n(Fonte: trn.json / AmazonTitles-1.3MM)\n")

if __name__ == "__main__":
    main()
