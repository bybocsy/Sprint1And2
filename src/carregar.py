
import pandas as pd

from config import ARQUIVOS, COLUNAS, COLUNAS_PESO, COLUNAS_TEXTO, DADOS_BRUTOS


def _corrigir_codificacao(texto):
    
    if not isinstance(texto, str) or texto.isascii():
        return texto
    try:
        return texto.encode("cp1252").decode("cp850")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def ler_arquivo(ano):
    
    caminho = DADOS_BRUTOS / ARQUIVOS[ano]

    df = pd.read_csv(caminho, skiprows=2, dtype=str, encoding="utf-8")
    df = df.iloc[:, :12]         
    df.columns = COLUNAS         

    if ano == 2024:
        for col in COLUNAS_TEXTO:
            valores = df[col].unique()
            mapa = {v: _corrigir_codificacao(v) for v in valores}
            df[col] = df[col].map(mapa)

    df["ARQUIVO"] = ano
    return df


def carregar_brutos(anos=None):

    anos = anos or list(ARQUIVOS)
    df = pd.concat([ler_arquivo(a) for a in anos], ignore_index=True)

    df["DATA"] = pd.to_datetime(df["DATA"], format="%d/%m/%Y")
    for col in COLUNAS_PESO:
        df[col] = pd.to_numeric(df[col]).astype("int64")
    for col in COLUNAS_TEXTO:
        df[col] = df[col].str.strip()
    return df


if __name__ == "__main__":
    base = carregar_brutos()
    print(base.shape)
    print(base.dtypes)
    print(base.memory_usage(deep=True).sum() / 1e6, "MB")
