from fastapi import FastAPI, APIRouter, HTTPException
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import json
import glob
import google.generativeai as genai
from openai import OpenAI
import re

# ================= CONFIGURAÇÃO =================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralForge")

# Pasta onde os arquivos JSON ficarão
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)

# Chaves
GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_KEYS", "").split(",") if k.strip()]
GROQ_KEY = os.environ.get("GROQ_KEY", "")

# Gerenciador Simples de Chaves
current_key_index = 0
def get_gemini_key():
    global current_key_index
    if not GEMINI_KEYS: raise Exception("Sem chaves Gemini")
    key = GEMINI_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_KEYS)
    return key

# ================= LÓGICA DO GEMINI "TERMINAL" =================
async def gemini_terminal_create_file(topic: str, count: int, filename: str):
    """
    Simula um terminal onde o Gemini deve executar o comando de criação de arquivo.
    """
    prompt = f"""
    ATUAR COMO: Um sistema Linux automatizado que gera datasets.
    TAREFA: Criar um arquivo JSON com {count} exemplos de treinamento sobre '{topic}'.
    
    ESTRUTURA DO JSON (Obrigatória):
    [
      {{
        "instruction": "pergunta do usuário",
        "input": "",
        "output": "resposta ideal"
      }}
    ]

    COMANDO RECEBIDO: 
    > cat > {filename} <<EOF
    [CONTEÚDO DO JSON AQUI]
    EOF

    SUA RESPOSTA:
    Apenas o conteúdo JSON puro que iria dentro do arquivo. Nada de markdown, nada de explicações.
    Comece com [ e termine com ].
    """
    
    try:
        api_key = get_gemini_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = await model.generate_content_async(prompt)
        content = response.text
        
        # Limpeza bruta para garantir que é JSON
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Validação: Tenta carregar para ver se é JSON válido
        json_content = json.loads(content)
        
        # Salva o arquivo fisicamente
        file_path = DATASET_DIR / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
            
        return len(json_content)
        
    except Exception as e:
        logger.error(f"Erro na geração: {e}")
        raise e

# ================= ALUNO (GROQ) =================
async def query_groq(prompt):
    if not GROQ_KEY: raise Exception("Sem chave Groq")
    client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

# ================= APP API =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")

class GenerateRequest(BaseModel):
    topic: str
    count: int = 1

class TestRequest(BaseModel):
    prompt: str

@api_router.get("/")
async def root():
    return {"status": "online", "system": "Terminal Simulator"}

@api_router.post("/generate")
async def generate(req: GenerateRequest):
    # Lógica para nomear arquivo: dataset.json -> dataset1.json -> dataset2.json
    existing = list(DATASET_DIR.glob("dataset*.json"))
    
    if not existing:
        filename = "dataset.json"
    else:
        # Encontra o maior número
        max_num = 0
        for f in existing:
            # Extrai número do nome (dataset5.json -> 5)
            match = re.search(r'dataset(\d*)\.json', f.name)
            if match:
                num_str = match.group(1)
                num = int(num_str) if num_str else 0 # dataset.json conta como 0
                if num > max_num: max_num = num
        
        filename = f"dataset{max_num + 1}.json"

    try:
        count = await gemini_terminal_create_file(req.topic, req.count, filename)
        return {"message": f"Arquivo criado: {filename}", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dataset")
async def list_data():
    all_items = []
    # Lê todos os arquivos da pasta
    files = sorted(DATASET_DIR.glob("*.json"))
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    # Adiciona nome do arquivo para referência
                    for item in data:
                        item['file'] = f.name
                    all_items.extend(data)
        except:
            continue
            
    return all_items

@api_router.delete("/dataset")
async def clear_data():
    files = DATASET_DIR.glob("*.json")
    for f in files:
        os.remove(f)
    return {"message": "Arquivos apagados"}

@api_router.post("/deepseek/test")
async def test_groq(req: TestRequest):
    try:
        res = await query_groq(req.prompt)
        return {"response": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)
