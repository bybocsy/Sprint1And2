import re
import unicodedata

import numpy as np
import pandas as pd

from config import CLASSES_ALVO, GRUPO_MATERIAL, SEMENTE


# ===================================================================== textos
def sem_acento(texto):
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _chave(texto):
    if not isinstance(texto, str):
        return texto
    texto = sem_acento(texto.upper().replace("–", "-"))
    return re.sub(r"\s+", " ", texto).strip()


NOMES_TRANSPORTADORA = {
    "SO ENTULHO TRANSPORTE DE CARGAS LTDA": "SO ENTULHOS TRANSPORTE DE CARGAS LTDA",
}

ROTULO_REGIONAL = {"MANUTENCAO": "MANUTENÇÃO"}


def padronizar_textos(df):
    df = df.copy()
    for col in ["ATERRO", "TRANSPORTADORA", "MATERIAL", "ZGL", "REGIONAL"]:
        valores = df[col].unique()
        df[col] = df[col].map({v: _chave(v) for v in valores})

    df["TRANSPORTADORA"] = df["TRANSPORTADORA"].replace(NOMES_TRANSPORTADORA)
    df["REGIONAL"] = df["REGIONAL"].replace(ROTULO_REGIONAL)

    chassi = df["CHASSI"].str.upper().str.replace("CAMINHAO.", "", regex=False)
    df["CHASSI"] = chassi.replace({"VEICULOS URBANO.CARGA": "OUTRO"})

    df["TIPOCAD"] = df["TIPOCAD"].str.capitalize()
    df["INATIVO"] = df["INATIVO"].str.capitalize()
    return df


def remover_duplicatas(df):
    return df.drop_duplicates(subset=[c for c in df.columns if c != "ARQUIVO"])


def remover_inativos(df):
    return df[df["INATIVO"] == "Nao"]


def remover_pesos_invalidos(df):
    ok = (
        (df["PESO_LIQUIDO"] > 0)
        & (df["TARA"] > 0)
        & (df["PESO_BRUTO"] - df["TARA"] == df["PESO_LIQUIDO"])
    )
    return df[ok]


def limites_iqr(serie, k=3.0):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def remover_outliers(df, k=3.0):
    
    manter = pd.Series(True, index=df.index)
    for _, g in df.groupby("CHASSI"):
        lo, hi = limites_iqr(g["TARA"], k)
        manter[g.index] &= g["TARA"].between(lo, hi)
    for _, g in df.groupby("MATERIAL_GRUPO"):
        lo, hi = limites_iqr(g["PESO_LIQUIDO"], k)
        manter[g.index] &= g["PESO_LIQUIDO"].between(lo, hi)
    return df[manter]


def origem_da_zona(zgl, regional):
    if zgl.startswith("CAU"):
        return "Caucaia"
    if regional.startswith("SER"):
        return "Fortaleza - coleta regular"
    return "Grandes geradores e outros"


def criar_atributos(df):
    df = df.copy()
    df["ANO"] = df["DATA"].dt.year
    df["MES"] = df["DATA"].dt.month
    df["DIA_SEMANA"] = df["DATA"].dt.dayofweek          # 0 = segunda
    df["FIM_DE_SEMANA"] = (df["DIA_SEMANA"] >= 5).astype(int)
    df["DIA_MES"] = df["DATA"].dt.day

    pares = df[["ZGL", "REGIONAL"]].drop_duplicates()
    pares["ORIGEM"] = [origem_da_zona(z, r) for z, r in pares.itertuples(index=False)]
    df = df.merge(pares, on=["ZGL", "REGIONAL"], how="left")

    df["MATERIAL_GRUPO"] = df["MATERIAL"].map(GRUPO_MATERIAL).fillna("Outros")

    df["CARGA_POR_TARA"] = df["PESO_LIQUIDO"] / df["TARA"]
    return df


ATRIBUTOS_NUMERICOS = ["TARA", "PESO_LIQUIDO", "CARGA_POR_TARA", "DIA_MES"]
ATRIBUTOS_CICLICOS = {"MES": 12, "DIA_SEMANA": 7}
ATRIBUTOS_CATEGORICOS = ["TRANSPORTADORA", "ZGL", "REGIONAL", "ORIGEM",
                         "CHASSI", "TIPOCAD"]
ATRIBUTOS_BINARIOS = ["FIM_DE_SEMANA"]
ALVO = "MATERIAL_GRUPO"


def agrupar_raros(serie, minimo=0.005, rotulo="OUTRAS"):
    freq = serie.value_counts(normalize=True)
    raros = freq[freq < minimo].index
    return serie.where(~serie.isin(raros), rotulo)


def selecionar_base_modelo(df):
    
    base = df[df[ALVO].isin(CLASSES_ALVO)].copy()
    base["TRANSPORTADORA"] = agrupar_raros(base["TRANSPORTADORA"])
    base["ZGL"] = agrupar_raros(base["ZGL"], minimo=0.001)
    base["REGIONAL"] = agrupar_raros(base["REGIONAL"], minimo=0.001)
    colunas = (ATRIBUTOS_NUMERICOS + list(ATRIBUTOS_CICLICOS)
               + ATRIBUTOS_CATEGORICOS + ATRIBUTOS_BINARIOS + ["ANO", ALVO])
    return base[colunas].reset_index(drop=True)


def dividir_treino_teste(df, prop_teste=0.3, semente=SEMENTE):
    teste = df.groupby(ALVO, group_keys=False).sample(frac=prop_teste,
                                                       random_state=semente)
    treino = df.drop(teste.index)
    return treino.sample(frac=1, random_state=semente), teste


def codificar(df, categorias=None):

    out = df[ATRIBUTOS_NUMERICOS + ATRIBUTOS_BINARIOS].astype(float).copy()
    for col, periodo in ATRIBUTOS_CICLICOS.items():
        angulo = 2 * np.pi * df[col] / periodo
        out[f"{col}_SEN"] = np.sin(angulo)
        out[f"{col}_COS"] = np.cos(angulo)

    if categorias is None:
        categorias = {c: sorted(df[c].unique()) for c in ATRIBUTOS_CATEGORICOS}
    for col, valores in categorias.items():
        cat = pd.Categorical(df[col], categories=valores)
        dummies = pd.get_dummies(cat, prefix=col, dtype="uint8")
        dummies.index = df.index
        out = pd.concat([out, dummies], axis=1)
    return out, categorias


class Padronizador:
    def __init__(self, colunas):
        self.colunas = colunas

    def ajustar(self, df):
        self.media = df[self.colunas].mean()
        self.desvio = df[self.colunas].std(ddof=0).replace(0, 1)
        return self

    def transformar(self, df):
        df = df.copy()
        df[self.colunas] = (df[self.colunas] - self.media) / self.desvio
        return df
