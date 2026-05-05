# Reconhecimento Facial PRO

Projeto de reconhecimento facial em Python usando OpenCV.

A arquitetura foi organizada para ficar limpa, fácil de testar e simples de evoluir.

## O que tem aqui

- Captura de rostos pela webcam
- Treinamento com LBPH
- Reconhecimento em tempo real
- Mapeamento persistente de nomes para IDs
- Validações para evitar treino sem imagens
- Logs básicos
- CLI principal
- Estrutura pronta para GitHub

## Estrutura

```txt
reconhecimento_facial_pro/
├── assets/
│   └── models/
│       ├── lbph_model.yml
│       └── labels.json
├── data/
│   └── dataset/
│       └── .gitkeep
├── scripts/
│   ├── capture.py
│   ├── detect.py
│   ├── recognize.py
│   └── train.py
├── src/
│   └── reconhecimento_facial/
│       ├── __init__.py
│       ├── camera.py
│       ├── cli.py
│       ├── config.py
│       ├── detector.py
│       ├── labels.py
│       ├── logger.py
│       ├── recognizer.py
│       └── trainer.py
├── tests/
│   └── test_labels.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Instalação

Crie e ative um ambiente virtual:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como testar

### 1. Testar se o projeto abre

```bash
python -m reconhecimento_facial.cli --help
```

Se aparecer a lista de comandos, a instalação básica está funcionando.

### 2. Testar detecção de rosto

```bash
python scripts/detect.py
```

A webcam deve abrir com um retângulo no rosto. Pressione `q` para sair.

### 3. Capturar imagens de uma pessoa

```bash
python scripts/capture.py --name Joao --samples 50
```

Isso cria imagens em:

```txt
data/dataset/Joao/
```

Recomendações:
- olhe para a câmera
- mova um pouco o rosto
- capture com boa iluminação
- evite óculos escuros
- use pelo menos 40 imagens por pessoa

### 4. Treinar o modelo

```bash
python scripts/train.py
```

O treinamento gera:

```txt
assets/models/lbph_model.yml
assets/models/labels.json
```

### 5. Rodar reconhecimento facial

```bash
python scripts/recognize.py
```

A webcam abre e mostra o nome da pessoa reconhecida. Pressione `q` para sair.

## Comandos pela CLI principal

Também dá para usar tudo pela CLI:

```bash
python -m reconhecimento_facial.cli detect
python -m reconhecimento_facial.cli capture --name Joao --samples 50
python -m reconhecimento_facial.cli train
python -m reconhecimento_facial.cli recognize
```

## Ajuste de confiança

No reconhecimento, o LBPH retorna uma distância. Quanto menor, melhor.

O valor padrão é:

```txt
70
```

Você pode ajustar:

```bash
python scripts/recognize.py --threshold 65
```

Se reconhecer gente errada, diminua o valor.  
Se marcar muita gente como desconhecida, aumente um pouco.

## Observações importantes

Este projeto usa `opencv-contrib-python`, não apenas `opencv-python`, porque o reconhecedor LBPH fica no módulo `cv2.face`.

O dataset não é enviado para o GitHub. Fotos de pessoas devem ficar fora do repositório por privacidade.

## Erros comuns

### `AttributeError: module 'cv2' has no attribute 'face'`

Instale a dependência correta:

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

### Webcam não abre

Teste outro índice de câmera:

```bash
python scripts/detect.py --camera 1
```

### Treino falha dizendo que não encontrou imagens

Capture imagens primeiro:

```bash
python scripts/capture.py --name Joao --samples 50
```

## Segurança e privacidade

Não sobe fotos reais no GitHub. A pasta `data/dataset` fica ignorada no `.gitignore`.


## Correção de segurança da webcam

Os comandos que usam câmera agora possuem:

- trava de instância única por câmera;
- uma única janela fixa por execução;
- limite de FPS para não sobrecarregar a máquina;
- fechamento seguro com `q` ou `ESC`.
Se alguma janela travar, feche pelo gerenciador de tarefas e apague os arquivos em `data/runtime/`.

## Autor
João Pedro da Costa Rezende
