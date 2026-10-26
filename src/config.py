"""Configurações gerais do projeto: caminhos, nomes de colunas e cores."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "reports" / "figures"
TABELAS = RAIZ / "reports" / "tables"

ARQUIVOS = {
    2022: "rs-2022.csv",
    2023: "rs-2023.csv",
    2024: "rs-2024.csv",
}


COLUNAS = [
    "ATERRO", "DATA", "TRANSPORTADORA", "MATERIAL", "ZGL", "REGIONAL",
    "TIPOCAD", "PESO_BRUTO", "TARA", "PESO_LIQUIDO", "CHASSI", "INATIVO",
]
COLUNAS_PESO = ["PESO_BRUTO", "TARA", "PESO_LIQUIDO"]  
COLUNAS_TEXTO = [c for c in COLUNAS if c not in COLUNAS_PESO + ["DATA"]]


GRUPO_MATERIAL = {
    "DOMICILIAR": "Domiciliar",
    "LIXO ESPECIAL URBANA": "Lixo especial urbano",
    "CAPINA": "Poda e capina",
    "PODACAO": "Poda e capina",
    "PODA TRITURADA": "Poda e capina",
    "TRONCO DE PODA": "Poda e capina",
    "COMPOSTAGEM - PODAS DE ARVORES": "Poda e capina",
    "ENTULHO": "Construção e entulho",
    "RCC - CLASSE A": "Construção e entulho",
    "RCC - CLASSE B": "Construção e entulho",
    "RESIDUOS INERTES": "Construção e entulho",
    "CLASSE IIA COMUM": "Classe IIA (grandes geradores)",
    "CLASSE IIA UMIDADE>20%": "Classe IIA (grandes geradores)",
    "VARRICAO": "Varrição",

    "TRANSBORDO": "Transbordo",
    "OUTROS": "Outros",
    "CHORUME": "Outros",
    "COLETA SELETIVA": "Outros",
    "COMPOSTAGEM - LODO DE ETE": "Outros",
    "COMPOSTAGEM - RESIDUO ORGANICO": "Outros",
    "COMPOSTAGEM - TERRA DE REFINARIA": "Outros",
}
CLASSES_ALVO = [
    "Domiciliar", "Lixo especial urbano", "Classe IIA (grandes geradores)",
    "Poda e capina", "Construção e entulho", "Varrição",
]

ROTULO_CURTO = {
    "Domiciliar": "Domiciliar",
    "Lixo especial urbano": "Lixo especial\nurbano",
    "Classe IIA (grandes geradores)": "Classe IIA\n(grandes ger.)",
    "Poda e capina": "Poda e\ncapina",
    "Construção e entulho": "Construção\ne entulho",
    "Varrição": "Varrição",
}

SEMENTE = 42

CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
         "#008300", "#6250d6", "#e34948"]
COR_CLASSE = dict(zip(CLASSES_ALVO, CORES))
AZUL_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"]
CINZA = "#8c8b86"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
