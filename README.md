# Fine-tuning FLAN-T5 para Mapeamento Título→Descrição

## 🎯 Descrição

Este projeto implementa fine-tuning do modelo FLAN-T5 para mapear perguntas sobre títulos de produtos para suas respectivas descrições.

## 🚀 Funcionalidades

- **Carregamento e limpeza** de dados JSON
- **Fine-tuning** do modelo google/flan-t5-base
- **Geração de pares** pergunta→resposta
- **Avaliação** com métrica ROUGE
- **Demonstração** de inferência

## 📊 Dataset

- **Fonte:** trn.json (Amazon Titles)
- **Formato:** JSON Lines
- **Colunas:** title, content
- **Tamanho:** ~2.2M registros

## 🛠️ Tecnologias

- **Python 3.12**
- **Transformers 4.55.2**
- **PyTorch**
- **Pandas**
- **Datasets (Hugging Face)**

## 📁 Estrutura

```
fine-tuning/
├── tech_challenge_train.py    # Script principal
├── data/                      # Dados de treinamento
├── outputs/                   # Modelo treinado
└── README.md
```

## 🚀 Como Usar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Treinamento
```bash
python tech_challenge_train.py
```

### 3. Testar Modelo
O script inclui demonstração automática após o treinamento.

## ⚙️ Configurações

- **Modelo Base:** google/flan-t5-base
- **Épocas:** 2
- **Batch Size:** 4
- **Learning Rate:** 3e-4
- **Max Input Length:** 512 tokens
- **Max Target Length:** 256 tokens

## 📈 Métricas

- **ROUGE-L** para avaliação de qualidade
- **Loss** durante treinamento
- **Gradiente norm** para estabilidade

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT.

