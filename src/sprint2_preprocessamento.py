
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import preprocessamento as pp
from carregar import carregar_brutos
from config import (CINZA, CLASSES_ALVO, CORES, DADOS_PROCESSADOS,
                    REMOVER_INATIVOS, ROTULO_CURTO, TABELAS, TEXTO_2)
from graficos import aplicar_estilo, eixo_milhar, fonte, num_br, salvar, subtitulo


class Registro:
    def __init__(self):
        self.linhas = []

    def __call__(self, etapa, antes, depois, motivo):
        self.linhas.append({"etapa": etapa, "linhas_antes": len(antes),
                            "removidas": len(antes) - len(depois),
                            "linhas_depois": len(depois), "motivo": motivo})
        print(f"  {etapa:<45} -{len(antes) - len(depois):>7}  -> {len(depois):>8}")
        return depois

    def tabela(self):
        return pd.DataFrame(self.linhas)


def grafico_etapas(etapas):
    d = etapas[etapas.removidas > 0]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.removidas, color=CORES[1], height=0.6)
    for yi, v, depois in zip(y, d.removidas, d.linhas_depois):
        ax.text(v + d.removidas.max() * 0.01, yi,
                f"-{num_br(v, 0)}   (restam {num_br(depois, 0)})",
                va="center", fontsize=8.5, color=TEXTO_2)
    ax.set_yticks(y, d.etapa)
    ax.set_xlim(0, d.removidas.max() * 1.45)
    eixo_milhar(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Linhas removidas")
    total_ini, total_fim = etapas.linhas_antes.iloc[0], etapas.linhas_depois.iloc[-1]
    ax.set_title("Quantas linhas cada etapa da limpeza removeu", pad=22)
    subtitulo(ax, f"De {num_br(total_ini, 0)} pesagens no arquivo para "
                  f"{num_br(total_fim, 0)} na base do modelo "
                  f"({num_br(total_fim / total_ini * 100)}% aproveitado).")
    fonte(fig)
    salvar(fig, "01_linhas_removidas", "sprint2")


def grafico_outliers(antes, depois):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, sharex=True)
    for ax, df, titulo in [(axes[0], antes, "Antes"), (axes[1], depois, "Depois")]:
        d = df[df.MATERIAL_GRUPO.isin(CLASSES_ALVO)]
        dados = [d.loc[d.MATERIAL_GRUPO == c, "PESO_LIQUIDO"] / 1000 for c in CLASSES_ALVO]
        ax.boxplot(dados, orientation="horizontal", widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor="#9ec5f4", edgecolor=CINZA),
                   medianprops=dict(color="black", linewidth=1.5),
                   whiskerprops=dict(color=CINZA), capprops=dict(color=CINZA),
                   flierprops=dict(marker="o", markersize=2, alpha=0.25,
                                   markerfacecolor=CINZA, markeredgecolor="none"))
        ax.set_yticks(range(1, 7), [ROTULO_CURTO[c].replace("\n", " ") for c in CLASSES_ALVO])
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xlabel("Peso líquido (t)")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Peso líquido por classe, antes e depois de tirar os extremos",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fonte(fig)
    fig.tight_layout()
    salvar(fig, "02_outliers_antes_depois", "sprint2")


def grafico_divisao(treino, teste):
    pt = treino[pp.ALVO].value_counts(normalize=True).reindex(CLASSES_ALVO) * 100
    pe = teste[pp.ALVO].value_counts(normalize=True).reindex(CLASSES_ALVO) * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(6)
    ax.bar(x - 0.2, pt, width=0.37, color=CORES[0], label=f"Treino ({num_br(len(treino), 0)})")
    ax.bar(x + 0.2, pe, width=0.37, color=CORES[1], label=f"Teste ({num_br(len(teste), 0)})")
    for xi, v in zip(x, pt):
        ax.text(xi, v + 0.8, f"{num_br(v)}%", ha="center", fontsize=8.5, color=TEXTO_2)
    ax.set_xticks(x, [ROTULO_CURTO[c] for c in CLASSES_ALVO], fontsize=8.5)
    ax.set_ylabel("% das linhas")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Treino e teste têm a mesma proporção de cada classe", pad=22)
    razao = pt.max() / pt.min()
    subtitulo(ax, "Divisão estratificada 70/30. As classes são desbalanceadas: "
                  f"Varrição tem ~{razao:.0f} vezes menos linhas que Domiciliar.")
    fonte(fig)
    salvar(fig, "03_divisao_treino_teste", "sprint2")


def grafico_padronizacao(antes, depois, colunas):
    nomes = {"TARA": "Tara (kg)", "PESO_LIQUIDO": "Peso líquido (kg)",
             "CARGA_POR_TARA": "Carga ÷ tara", "DIA_MES": "Dia do mês"}
    fig, axes = plt.subplots(2, len(colunas), figsize=(12, 5))
    for j, c in enumerate(colunas):
        for i, (df, rot) in enumerate([(antes, "original"), (depois, "padronizado")]):
            ax = axes[i, j]
            ax.hist(df[c], bins=50, color=CORES[i], edgecolor="#fcfcfb", linewidth=0.4)
            ax.set_yticks([])
            ax.set_title(f"{nomes[c]} - {rot}", fontsize=9)
    fig.suptitle("Padronização z-score: mesma forma, nova escala (média 0, desvio 1)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fonte(fig)
    fig.tight_layout()
    salvar(fig, "04_padronizacao", "sprint2")


def main():
    aplicar_estilo()
    DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    TABELAS.mkdir(parents=True, exist_ok=True)
    reg = Registro()

    print("\n[1] Leitura e limpeza das linhas")
    df = carregar_brutos()
    df0 = df
    df = reg("Leitura dos 3 arquivos", df0, df, "Nenhuma linha removida")
    df = reg("Padronização dos textos", df, pp.padronizar_textos(df),
             "Codificação de 2024, acentos, traços, nomes de empresa, chassi")
    df = reg("Linhas repetidas", df, pp.remover_duplicatas(df),
             "Meses exportados 2 ou 3 vezes no arquivo de 2024")
    if REMOVER_INATIVOS:
        df = reg("INATIVO = Sim", df, pp.remover_inativos(df), "Pesagens inativas")
    else:
        df = reg("INATIVO = Sim (mantidas)", df, df,
                 "Parecem viagens reais de veículos desativados depois")
    df = reg("Pesos impossíveis ou com conta errada", df, pp.remover_pesos_invalidos(df),
             "Líquido <= 0, tara = 0 ou bruto - tara != líquido")

    print("\n[2] Novos atributos")
    df = pp.criar_atributos(df)
    antes_outliers = df
    df = reg("Valores extremos (3 x IQR)", df, pp.remover_outliers(df),
             "Tara extrema para o tipo de caminhão ou carga extrema para o material")
    grafico_outliers(antes_outliers, df)

    df.drop(columns=["INATIVO"]).to_csv(DADOS_PROCESSADOS / "base_limpa.csv.gz",
                                        index=False, compression="gzip")
    print("  salvo: data/processed/base_limpa.csv.gz")

    print("\n[3] Base do modelo")
    base = pp.selecionar_base_modelo(df)
    base = reg("Fora da variável-alvo", df, base,
               "Transbordo (lixo já pesado antes) e Outros (grupo misturado e pequeno)")
    etapas = reg.tabela()
    etapas.to_csv(TABELAS / "s2_etapas.csv", index=False, encoding="utf-8-sig")
    grafico_etapas(etapas)

    resumo_cat = pd.DataFrame({
        "valores_distintos": [base[c].nunique() for c in pp.ATRIBUTOS_CATEGORICOS]},
        index=pp.ATRIBUTOS_CATEGORICOS)
    resumo_cat.to_csv(TABELAS / "s2_categoricos.csv", encoding="utf-8-sig")
    print(resumo_cat)

    print("\n[4] Divisão treino/teste, codificação e padronização")
    treino, teste = pp.dividir_treino_teste(base)
    grafico_divisao(treino, teste)

    X_tr, categorias = pp.codificar(treino)
    X_te, _ = pp.codificar(teste, categorias)
    print(f"  atributos depois da codificação: {X_tr.shape[1]}")

    padr = pp.Padronizador(pp.ATRIBUTOS_NUMERICOS).ajustar(X_tr)
    Z_tr, Z_te = padr.transformar(X_tr), padr.transformar(X_te)
    grafico_padronizacao(X_tr, Z_tr, pp.ATRIBUTOS_NUMERICOS)
    pd.DataFrame({"media_treino": padr.media, "desvio_treino": padr.desvio}).round(3) \
        .to_csv(TABELAS / "s2_padronizacao.csv", encoding="utf-8-sig")

    grupos = {"Numéricos padronizados": pp.ATRIBUTOS_NUMERICOS,
              "Cíclicos (seno/cosseno)": [c for c in X_tr if c.endswith(("_SEN", "_COS"))],
              "Binário": pp.ATRIBUTOS_BINARIOS}
    for c in pp.ATRIBUTOS_CATEGORICOS:
        grupos[f"One-hot de {c}"] = [k for k in X_tr if k.startswith(c + "_")]
    pd.Series({k: len(v) for k, v in grupos.items()}, name="colunas") \
        .to_csv(TABELAS / "s2_atributos_finais.csv", encoding="utf-8-sig")

    for nome, Z, parte in [("treino", Z_tr, treino), ("teste", Z_te, teste)]:
        saida = Z.assign(ANO=parte["ANO"].values, ALVO=parte[pp.ALVO].values)
        saida.to_csv(DADOS_PROCESSADOS / f"{nome}.csv.gz", index=False,
                     compression="gzip", float_format="%.5f")
        print(f"  salvo: data/processed/{nome}.csv.gz  {saida.shape}")

    # checagens finais
    assert not Z_tr.isna().any().any() and not Z_te.isna().any().any()
    assert list(Z_tr.columns) == list(Z_te.columns)
    assert np.allclose(Z_tr[pp.ATRIBUTOS_NUMERICOS].mean(), 0, atol=1e-9)
    print("\nPronto.")


if __name__ == "__main__":
    main()
