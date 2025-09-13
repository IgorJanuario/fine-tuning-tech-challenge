import os
import random
import json
import pandas as pd
import torch
import gc
from typing import Optional, Dict, Any
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


SEED = 42
random.seed(SEED)

# Configurações padrão
TRN_JSON_PATH = "data/trn.json"
OUTPUT_DIR = "outputs/flan_t5_base_tech_challenge"
BASE_MODEL = "google/flan-t5-base"

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 256
MAX_SAMPLES = None

# Configurações para treinamento em chunks
CHUNK_SIZE = 4000  # Número de registros por chunk
MAX_SAMPLES_PER_CHUNK = None  # Limite opcional por chunk
CHUNKED = True # Flag para ativar treinamento em chunks
START_CHUNK = 0
END_CHUNK = None

# Configurações de otimização de memória CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Funções para gerenciamento de GPU
def print_gpu_info():
    """Imprime informações detalhadas sobre a GPU"""
    if not torch.cuda.is_available():
        print("❌ GPU não disponível, usando CPU")
        return

    print("✅ GPU disponível!")
    print(f"📊 Dispositivos GPU encontrados: {torch.cuda.device_count()}")
    print(f"🎯 Dispositivo atual: {torch.cuda.current_device()}")

    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
        free_memory = total_memory - reserved_memory

        print(f"\n🖥️  GPU {i}: {device_name}")
        print(f"   💾 Memória total: {total_memory:.2f} GB")
        print(f"   📈 Memória alocada: {allocated_memory:.2f} GB")
        print(f"   🔒 Memória reservada: {reserved_memory:.2f} GB")
        print(f"   🆓 Memória livre: {free_memory:.2f} GB")

def clear_gpu_cache():
    """Limpa o cache da GPU para liberar memória"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Cache da GPU limpo")

def monitor_gpu_memory():
    """Monitora o uso de memória da GPU em tempo real"""
    if not torch.cuda.is_available():
        print("GPU não disponível para monitoramento")
        return

    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    free = total - reserved

    print(f"📊 GPU {current_device} - Alocada: {allocated:.2f}GB | Reservada: {reserved:.2f}GB | Livre: {free:.2f}GB | Total: {total:.2f}GB")
    
    # Aviso se memória livre está baixa
    if free < 10.0:
        print(f"⚠️  ATENÇÃO: Apenas {free:.2f}GB livres! Considere reduzir batch size ou chunk size.")
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }

def check_memory_requirements(chunk_size: int, batch_size: int = 4) -> bool:
    """Verifica se há memória suficiente para o treinamento"""
    if not torch.cuda.is_available():
        return True
    
    memory_info = monitor_gpu_memory()
    if not memory_info:
        return False
    
    # Estimativa grosseira de memória necessária
    # Cada registro gera ~3 pares, cada par usa ~2KB tokenizado
    estimated_pairs = chunk_size * 3
    estimated_memory_gb = (estimated_pairs * 2 * batch_size) / (1024**3) * 0.001  # Conversão aproximada
    
    print(f"📊 Estimativa de memória necessária: {estimated_memory_gb:.2f}GB")
    print(f"📊 Memória livre disponível: {memory_info['free']:.2f}GB")
    
    if memory_info['free'] < estimated_memory_gb + 10:  # 10GB de margem
        print(f"⚠️  ATENÇÃO: Memória insuficiente! Considere reduzir chunk_size ou batch_size")
        return False
    
    return True

def setup_device():
    """Configura e retorna o dispositivo apropriado (GPU ou CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_gpu_info()
    else:
        device = torch.device("cpu")
        print("❌ GPU não disponível, usando CPU")

    return device

def find_optimal_batch_size(model, tokenizer, device, max_batch_size=16):
    """Encontra o batch size ótimo para a GPU disponível"""
    if not torch.cuda.is_available():
        return 2  # Batch size conservador para CPU

    print("🔍 Detectando batch size ótimo para GPU...")

    # Dados de teste simples
    test_input = "What is this product: Test Product?"
    test_target = "This is a test product description."

    optimal_batch_size = 1
    for batch_size in [1, 2, 4, 8, 16]:
        if batch_size > max_batch_size:
            break

        try:
            # Criar batch de teste
            inputs = [test_input] * batch_size
            targets = [test_target] * batch_size

            # Tokenizar
            model_inputs = tokenizer(
                inputs,
                max_length=MAX_INPUT_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=MAX_TARGET_LEN,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )

            model_inputs["labels"] = labels["input_ids"].to(device)

            # Testar forward pass
            with torch.no_grad():
                outputs = model(**model_inputs)

            # Limpar cache
            torch.cuda.empty_cache()
            gc.collect()

            print(f"✅ Batch size {batch_size}: OK")
            optimal_batch_size = batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Batch size {batch_size}: Out of memory")
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise e

    print(f"🎯 Batch size ótimo detectado: {optimal_batch_size}")
    return optimal_batch_size

# Classes para treinamento em chunks
class ChunkedDataLoader:
    """Carregador de dados que processa arquivos grandes em chunks"""
    
    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.total_lines = self._count_lines()
        
    def _count_lines(self) -> int:
        """Conta o número total de linhas no arquivo"""
        print(f"📊 Contando linhas em {self.file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"✅ Total de linhas: {count:,}")
        return count
    
    def load_chunk(self, chunk_index: int, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Carrega um chunk específico do arquivo"""
        start_line = chunk_index * self.chunk_size
        end_line = min(start_line + self.chunk_size, self.total_lines)
        
        print(f"📦 Carregando chunk {chunk_index + 1}: linhas {start_line + 1:,} a {end_line:,}")
        
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Pular até a linha inicial
            for _ in range(start_line):
                next(f, None)
            
            # Ler as linhas do chunk
            for line_num in range(start_line, end_line):
                line = f.readline()
                if not line:
                    break
                    
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Erro na linha {line_num + 1}: {e}")
                    continue
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Filtrar colunas necessárias
        keep = [c for c in df.columns if c in ("title", "content")]
        df = df[keep].dropna(subset=["title", "content"])
        
        # Limpar dados
        df["title"] = df["title"].astype(str).str.strip()
        df["content"] = df["content"].astype(str).str.strip()
        df = df[(df["title"].str.len() > 2) & (df["content"].str.len() > 10)]
        
        # Limitar amostras se especificado
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=SEED)
            
        df = df.reset_index(drop=True)
        print(f"✅ Chunk {chunk_index + 1} carregado: {len(df):,} registros válidos")
        
        return df
    
    def get_total_chunks(self) -> int:
        """Retorna o número total de chunks"""
        return (self.total_lines + self.chunk_size - 1) // self.chunk_size
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os chunks"""
        total_chunks = self.get_total_chunks()
        return {
            "total_lines": self.total_lines,
            "chunk_size": self.chunk_size,
            "total_chunks": total_chunks,
            "estimated_records_per_chunk": self.chunk_size,
        }

class MemoryCallback(TrainerCallback):
    """Callback para monitorar e gerenciar memória durante o treinamento"""
    
    def __init__(self, cleanup_frequency: int = 50):
        self.cleanup_frequency = cleanup_frequency
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        
        if self.step_count % self.cleanup_frequency == 0:
            if torch.cuda.is_available():
                # Usar a função de monitoramento corrigida
                memory_info = monitor_gpu_memory()
                
                # Limpar cache se memória livre < 10GB
                if memory_info and memory_info["free"] < 10.0:
                    print("🧹 Limpando cache da GPU...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Verificar memória após limpeza
                    memory_after = monitor_gpu_memory()
                    if memory_after:
                        freed = memory_info["free"] - memory_after["free"]
                        print(f"🧹 Memória liberada: {freed:.2f}GB")

def load_trn(trn_path: str) -> pd.DataFrame:
    data = []
    with open(trn_path, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num + 1} due to JSON decoding error: {e}")

    df = pd.DataFrame(data)
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

def train_chunked(file_path: str, chunk_size: int = 10000, 
                 max_samples_per_chunk: Optional[int] = None,
                 start_chunk: int = 0, end_chunk: Optional[int] = None,
                 device=None, base_model: str = BASE_MODEL, output_dir: str = OUTPUT_DIR):
    """Treina modelos separados para cada chunk e depois os incrementa em um modelo final"""
    print(f"🎯 Iniciando treinamento chunked")
    print(f"📁 Arquivo: {file_path}")
    print(f"📦 Tamanho do chunk: {chunk_size:,}")
    print(f"🔄 Modo: Salvar todos os chunks + Incrementar em modelo final")
    
    # Configurar dispositivo
    if device is None:
        device = setup_device()
    
    # Carregar modelo e tokenizer base
    print(f"📥 Carregando modelo base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Configurar carregador de chunks
    chunk_loader = ChunkedDataLoader(file_path, chunk_size)
    chunk_info = chunk_loader.get_chunk_info()
    
    print(f"📊 Informações dos chunks:")
    print(f"   Total de linhas: {chunk_info['total_lines']:,}")
    print(f"   Total de chunks: {chunk_info['total_chunks']}")
    print(f"   Tamanho do chunk: {chunk_info['chunk_size']:,}")
    
    # Determinar range de chunks
    if end_chunk is None:
        end_chunk = chunk_info['total_chunks']
    
    print(f"🔄 Processando chunks {start_chunk + 1} a {end_chunk}")
    
    # Verificar requisitos de memória
    print("🔍 Verificando requisitos de memória...")
    if not check_memory_requirements(chunk_size):
        print("❌ Memória insuficiente! Reduzindo chunk_size automaticamente...")
        chunk_size = min(chunk_size // 2, 10000)  # Reduzir pela metade, máximo 10k
        print(f"📦 Novo chunk_size: {chunk_size}")
        
        # Recriar carregador com novo tamanho
        chunk_loader = ChunkedDataLoader(file_path, chunk_size)
        chunk_info = chunk_loader.get_chunk_info()
        end_chunk = chunk_info['total_chunks']
        print(f"📊 Total de chunks atualizado: {chunk_info['total_chunks']}")
    
    # Lista para armazenar caminhos dos modelos de chunks
    chunk_model_paths = []
    
    # Processar cada chunk
    for chunk_index in range(start_chunk, end_chunk):
        try:
            # Carregar chunk
            df_chunk = chunk_loader.load_chunk(chunk_index, max_samples_per_chunk)
            
            if df_chunk.empty:
                print(f"⚠️  Chunk {chunk_index + 1} vazio, pulando...")
                continue
            
            print(f"\n🚀 Iniciando treinamento no chunk {chunk_index + 1}")
            print(f"📊 Registros no chunk: {len(df_chunk):,}")
            
            # Preparar dataset
            df_pairs = build_supervised_pairs(df_chunk)
            print(f"📈 Pares gerados: {len(df_pairs):,}")
            
            dataset = to_hf_dataset(df_pairs)
            
            # Tokenizar
            print("🔤 Tokenizando dataset...")
            tokenized_train = dataset["train"].map(
                lambda x: tokenize_function(x, tokenizer), 
                batched=True, 
                remove_columns=dataset["train"].column_names
            )
            tokenized_val = dataset["validation"].map(
                lambda x: tokenize_function(x, tokenizer), 
                batched=True, 
                remove_columns=dataset["validation"].column_names
            )
            
            # Configurar diretório de saída para este chunk
            chunk_output_dir = os.path.join(output_dir, f"chunk_{chunk_index + 1}")
            os.makedirs(chunk_output_dir, exist_ok=True)
            
            # Carregar modelo base para este chunk (cada chunk treina um modelo independente)
            print(f"📥 Carregando modelo base para chunk {chunk_index + 1}")
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
            model.gradient_checkpointing_enable()
            
            # Detectar batch size ótimo
            optimal_batch_size = find_optimal_batch_size(model, tokenizer, device)
            
            # Configurar argumentos de treinamento
            training_args = TrainingArguments(
                output_dir=chunk_output_dir,
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=500,
                save_steps=500,
                learning_rate=3e-4,
                per_device_train_batch_size=optimal_batch_size,
                per_device_eval_batch_size=optimal_batch_size,
                gradient_accumulation_steps=1,
                num_train_epochs=2,
                weight_decay=0.01,
                logging_steps=50,
                warmup_steps=100,
                report_to="none",
                remove_unused_columns=False,
                include_inputs_for_metrics=True,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=2 if torch.cuda.is_available() else 0,
                dataloader_pin_memory=torch.cuda.is_available(),
                gradient_checkpointing=True,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
            )
            
            # Configurar data collator
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            
            # Configurar callbacks
            callbacks = [MemoryCallback()]
            
            # Criar trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda p: compute_rouge(p, tokenizer),
                callbacks=callbacks,
            )
            
            # Treinar
            print("🏋️  Iniciando treinamento...")
            print("📊 Memória antes do treinamento:")
            memory_before = monitor_gpu_memory()
            
            trainer.train()
            
            print("📊 Memória após o treinamento:")
            memory_after = monitor_gpu_memory()
            
            if memory_before and memory_after:
                memory_used = memory_after["allocated"] - memory_before["allocated"]
                print(f"📈 Memória adicional usada durante treinamento: {memory_used:.2f}GB")
            
            # Salvar modelo do chunk
            model_path = os.path.join(chunk_output_dir, "final_model")
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            chunk_model_paths.append(model_path)
            print(f"💾 Modelo do chunk {chunk_index + 1} salvo em: {model_path}")
            
            # Avaliar
            print("📊 Avaliando modelo...")
            eval_results = trainer.evaluate()
            print(f"📈 Resultados do chunk {chunk_index + 1}: {eval_results}")
            
            # Limpar memória
            print("🧹 Limpando memória após chunk...")
            del trainer, tokenized_train, tokenized_val, dataset, df_pairs, df_chunk, model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            # Verificar memória após limpeza
            print("📊 Memória após limpeza:")
            memory_after_cleanup = monitor_gpu_memory()
            
            if memory_after and memory_after_cleanup:
                memory_freed = memory_after["allocated"] - memory_after_cleanup["allocated"]
                print(f"🧹 Memória liberada na limpeza: {memory_freed:.2f}GB")
            
            print(f"✅ Chunk {chunk_index + 1} concluído")
            
        except Exception as e:
            print(f"❌ Erro no chunk {chunk_index + 1}: {e}")
            print("🔄 Continuando com o próximo chunk...")
            continue
    
    # Incrementar todos os modelos de chunks em um modelo final
    if chunk_model_paths:
        print(f"\n🔄 Incrementando {len(chunk_model_paths)} modelos de chunks em modelo final...")
        final_model_path = increment_models_from_chunks(
            chunk_model_paths=chunk_model_paths,
            base_model=base_model,
            output_dir=output_dir,
            device=device,
            tokenizer=tokenizer
        )
        print(f"🎉 Treinamento chunked concluído!")
        print(f"🏆 Modelo final unificado salvo em: {final_model_path}")
        return final_model_path
    else:
        print("❌ Nenhum modelo de chunk foi treinado com sucesso")
        return None

def increment_models_from_chunks(chunk_model_paths: list, base_model: str, output_dir: str, 
                                device, tokenizer) -> str:
    """Incrementa todos os modelos de chunks em um modelo final unificado"""
    print(f"🔄 Iniciando incremento de {len(chunk_model_paths)} modelos...")
    
    # Carregar modelo base
    print(f"📥 Carregando modelo base: {base_model}")
    final_model = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    final_model.gradient_checkpointing_enable()
    
    # Configurar diretório final
    final_model_path = os.path.join(output_dir, "final_unified_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # Processar cada modelo de chunk
    for i, chunk_model_path in enumerate(chunk_model_paths):
        try:
            print(f"\n🔄 Processando modelo do chunk {i + 1}/{len(chunk_model_paths)}")
            print(f"📁 Caminho: {chunk_model_path}")
            
            # Carregar modelo do chunk
            chunk_model = AutoModelForSeq2SeqLM.from_pretrained(chunk_model_path).to(device)
            
            # Incrementar pesos do modelo final com os pesos do chunk
            # Estratégia: Média ponderada dos pesos (peso igual para todos os chunks)
            weight_factor = 1.0 / len(chunk_model_paths)
            
            with torch.no_grad():
                for (name, final_param), (_, chunk_param) in zip(final_model.named_parameters(), chunk_model.named_parameters()):
                    if final_param.shape == chunk_param.shape:
                        # Incrementar com média ponderada
                        final_param.data = final_param.data + (chunk_param.data - final_param.data) * weight_factor
                    else:
                        print(f"⚠️  Parâmetro {name} tem shapes diferentes: {final_param.shape} vs {chunk_param.shape}")
            
            # Limpar memória
            del chunk_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            print(f"✅ Chunk {i + 1} incrementado com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao processar chunk {i + 1}: {e}")
            continue
    
    # Salvar modelo final
    print(f"💾 Salvando modelo final unificado...")
    final_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Limpar memória
    del final_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    print(f"✅ Modelo final unificado salvo em: {final_model_path}")
    return final_model_path

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configurar dispositivo e mostrar informações da GPU
    device = setup_device()

    # Verificar se deve usar treinamento em chunks
    if CHUNKED:
        print("🚀 Modo: Treinamento em Chunks")
        print(f"📁 Arquivo: {TRN_JSON_PATH}")
        print(f"📦 Tamanho do chunk: {CHUNK_SIZE:,}")
        
        # Verificar se arquivo existe
        if not os.path.exists(TRN_JSON_PATH):
            print(f"❌ Arquivo não encontrado: {TRN_JSON_PATH}")
            return
        
        # Treinar em chunks
        model_path = train_chunked(
            file_path=TRN_JSON_PATH,
            chunk_size=CHUNK_SIZE,
            max_samples_per_chunk=MAX_SAMPLES_PER_CHUNK,
            start_chunk=START_CHUNK,
            end_chunk=END_CHUNK,
            device=device,
            base_model=BASE_MODEL,
            output_dir=OUTPUT_DIR
        )
        
        # Carregar modelo final para demonstração
        if model_path and os.path.exists(model_path):
            print(f"\n🎯 Carregando modelo final para demonstração...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            
            # Demonstração com alguns títulos de exemplo
            demo_titles = ["iPhone 15", "Samsung Galaxy S24", "MacBook Pro M3"]
            questions = [f"What is this product: {t}?" for t in demo_titles]
            print("\n=== DEMONSTRAÇÃO ===")
            for q in questions:
                prompt = f"Answer the user question using the product description.\nQuestion: {q}\n"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(device)
                outputs = model.generate(**inputs, max_new_tokens=192)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"P: {q}\nR: {answer}\n")
        
        return

    # Treinamento tradicional (modo original)
    print("🚀 Modo: Treinamento Tradicional")
    print("Carregando trn.json ...")
    df_raw = load_trn(TRN_JSON_PATH)
    quick_check(df_raw)
    print(f"Registros válidos: {len(df_raw)}")

    print("Gerando pares (pergunta sobre título -> descrição)...")
    df_pairs = build_supervised_pairs(df_raw)
    print(f"Total de pares gerados: {len(df_pairs)}")

    dsets = to_hf_dataset(df_pairs)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    # Detectar batch size ótimo para GPU
    optimal_batch_size = find_optimal_batch_size(model, tokenizer, device)

    tok_train = dsets["train"].map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dsets["train"].column_names)
    tok_val   = dsets["validation"].map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dsets["validation"].column_names)
    tok_test  = dsets["test"].map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dsets["test"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args_training = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=optimal_batch_size,
        per_device_eval_batch_size=optimal_batch_size,
        weight_decay=0.01,
        num_train_epochs=2,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        include_inputs_for_metrics=True,
        # Otimizações para GPU
        dataloader_pin_memory=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),  # Usar precisão mista se GPU disponível
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
    )

    trainer = Trainer(
        model=model,
        args=args_training,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_rouge(p, tokenizer),
    )

    print("Treinando...")
    monitor_gpu_memory()  # Monitorar antes do treinamento
    trainer.train()
    monitor_gpu_memory()  # Monitorar após o treinamento

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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(device)
        outputs = model.generate(**inputs, max_new_tokens=192)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"P: {q}\nR: {answer}\n(Fonte: trn.json / AmazonTitles-1.3MM)\n")


if __name__ == "__main__":
    main()