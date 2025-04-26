# Projeto Final de Deep Learning

Detecção e Segmentação de Semáforos de trânsito !

## Introdução

Projeto Final da Disciplina de Deep Learning

---

## Requisitos do Sistema

### Python

Este projeto requer Python 3.12.3. Para verificar sua versão do Python, execute:

```bash
python --version
```

Se você não tem a versão correta, aqui estão as opções para instalação:

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12
```

#### MacOS (usando Homebrew)

```bash
brew install python@3.12
```

#### Windows

1. Baixe o instalador do Python 3.12.3 em https://www.python.org/downloads/
2. Execute o instalador
3. Marque a opção "Add Python 3.12 to PATH"
4. Clique em "Install Now"

## Instalação do Projeto

1. Clone o repositório:

```bash
git clone git@github.com:erlonL/dl-projeto-final.git
cd dl-projeto-final
```

2. Instale o gerenciador de pacotes `uv`:

Siga as instruções na [documentação oficial do uv](https://docs.astral.sh/uv/) para instalar o `uv`.

3. Instale as dependências do projeto:

```bash
uv sync
```

4. (Opcional) Crie um kernel para rodar os Jupyter Notebooks:

Primeiro, adicione o `ipykernel` como dependência de desenvolvimento:

```bash
uv add --dev ipykernel
```

Depois, crie o kernel:

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=dl-projeto-final
```

## Como Usar

O projeto pode ser executado de duas formas:

1. Usando o `uv`:

```bash
uv run example.py
```

2. Ativando o ambiente virtual manualmente e rodando o script diretamente:

```bash
source .venv/bin/activate  # Linux/MacOS
# ou
.venv\Scripts\activate     # Windows

python example.py
```

## Abordagem

1. Primeiro detectar onde está o objeto na imagem
2. Transformar a imagem para ficar "flat" (spatial transformer)
3. Detectar qual objeto é (verde, vermelho, amarelo)

## Motivações

1. Carros autômatos
2. Redução de escopo (apenas semáforo)
3. Semáforo é uma sinalização universal

## Dataset

- Redução do dataset do artigo br
- Vídeos do youtube
- 3 classes: verde, vermelho, amarelo
- outras fontes

## Arquitetura

- abordagem parecida com a do artigo
- transfer learning
- mask CNN

## Links

https://journals-sol.sbc.org.br/index.php/jbcs/article/view/3678/2794  
https://arxiv.org/abs/1703.06870  
https://github.com/xuexingyu24/License_Plate_Detection_Pytorch  
Region Proposal Network (RPN)
