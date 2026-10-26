import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from carregar import carregar_brutos
from config import CINZA, CLASSES_ALVO, CORES, ROTULO_CURTO, TABELAS, TEXTO, TEXTO_2
from graficos import (MAPA_DIVERGENTE, MAPA_SEQUENCIAL, aplicar_estilo,
                      eixo_datas, eixo_milhar, fonte, num_br, salvar, subtitulo)
from preprocessamento import (criar_atributos, padronizar_textos,
                              remover_duplicatas, remover_pesos_invalidos)

DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
COR_ORIGEM = {"Fortaleza - coleta regular": CORES[0],
              "Grandes geradores e outros": CORES[1],
              "Caucaia": CORES[2]}


def tabela(df, nome):
    TABELAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABELAS / f"{nome}.csv", encoding="utf-8-sig")
    print("  tabela salva: reports/tables/" + nome + ".csv")


def descrever_base(brutos):
    print("\n[1] Descrição da base")
    tipos = {
        "ATERRO": "Categórico (nominal)", "DATA": "Data",
        "TRANSPORTADORA": "Categórico (nominal)", "MATERIAL": "Categórico (nominal)",
        "ZGL": "Categórico (nominal)", "REGIONAL": "Categórico (nominal)",
        "TIPOCAD": "Categórico (binário)", "PESO_BRUTO": "Numérico (inteiro, kg)",
        "TARA": "Numérico (inteiro, kg)", "PESO_LIQUIDO": "Numérico (inteiro, kg)",
        "CHASSI": "Categórico (nominal)", "INATIVO": "Categórico (binário)",
    }
    resumo = pd.DataFrame({
        "tipo": pd.Series(tipos),
        "valores_ausentes": brutos[list(tipos)].isna().sum(),
        "valores_distintos": brutos[list(tipos)].nunique(),
    })
    tabela(resumo, "s1_atributos")

    por_arquivo = brutos.groupby("ARQUIVO").agg(
        linhas=("DATA", "size"), inicio=("DATA", "min"), fim=("DATA", "max"))
    por_arquivo["linhas_repetidas"] = brutos.duplicated().groupby(brutos.ARQUIVO).sum()
    por_arquivo["inativo_sim"] = brutos[brutos.INATIVO == "Sim"].groupby("ARQUIVO").size()
    tabela(por_arquivo.fillna(0), "s1_arquivos")
    print(por_arquivo)

    num = brutos[["PESO_BRUTO", "TARA", "PESO_LIQUIDO"]].describe().T
    tabela(num.round(1), "s1_estatisticas_pesos")


def grafico_registros_por_mes(brutos, limpos):
    print("\n[2] Pesagens por mês: arquivo x sem linhas repetidas")
    a = brutos.set_index("DATA").resample("MS").size()
    b = limpos.set_index("DATA").resample("MS").size()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(a.index, a.values, color=CORES[1], label="Como veio no arquivo")
    ax.plot(b.index, b.values, color=CORES[0], label="Sem linhas repetidas")
    for mes in ["2024-02-01", "2024-03-01", "2024-06-01"]:
        v = a.loc[mes]
        ax.annotate(f"{MESES[pd.Timestamp(mes).month - 1]}/24: {num_br(v, 0)}",
                    (pd.Timestamp(mes), v), xytext=(-8, 4), ha="right",
                    textcoords="offset points", fontsize=8.5, color=TEXTO_2)
    ax.set_ylim(0, a.max() * 1.12)
    eixo_milhar(ax)
    ax.set_ylabel("Pesagens no mês")
    ax.set_title("O arquivo de 2024 tem meses repetidos", pad=22)
    subtitulo(ax, "Fevereiro e junho aparecem 2 vezes e março, 3 vezes. "
                  "Em 2022 e 2023 são só 52 linhas repetidas.")
    ax.legend(loc="lower left")
    eixo_datas(ax)
    fonte(fig)
    salvar(fig, "01_pesagens_por_mes")


def grafico_inativos(brutos):
    print("\n[3] Registros com INATIVO = Sim")
    m = brutos.assign(MES=brutos.DATA.dt.to_period("M").dt.to_timestamp())
    prop = m.groupby("MES").INATIVO.apply(lambda s: (s == "Sim").mean() * 100)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(prop.index, prop.values, width=24, color=CORES[0])
    ax.set_ylabel("% das pesagens do mês")
    ax.set_title("Pesagens marcadas como INATIVO = Sim", pad=22)
    subtitulo(ax, "O arquivo de 2024 foi exportado só com registros ativos, "
                  "por isso fica zerado.")
    eixo_datas(ax)
    fonte(fig)
    salvar(fig, "02_inativos_por_mes")


def grafico_histogramas(df):
    print("\n[4] Histogramas dos pesos")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, col, nome in zip(axes, ["PESO_BRUTO", "TARA", "PESO_LIQUIDO"],
                             ["Peso bruto", "Tara (caminhão vazio)", "Peso líquido (carga)"]):
        t = df[col] / 1000
        ax.hist(t, bins=np.arange(0, 50.5, 1), color=CORES[0],
                edgecolor="#fcfcfb", linewidth=0.8)
        ax.axvline(t.median(), color=TEXTO, linewidth=1, linestyle="--")
        ax.text(t.median() + 0.8, ax.get_ylim()[1] * 0.92,
                f"mediana\n{num_br(t.median())} t", fontsize=8, color=TEXTO_2)
        ax.set_title(nome, fontsize=10.5)
        ax.set_xlabel("toneladas")
    eixo_milhar(axes[0])
    axes[0].set_ylabel("Número de pesagens")
    fig.suptitle("Como os pesos se distribuem", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fonte(fig)
    fig.tight_layout()
    salvar(fig, "03_histogramas_pesos")


def grafico_classes(df):
    print("\n[5] Distribuição das classes (material agrupado)")
    ordem = CLASSES_ALVO + ["Transbordo", "Outros"]
    viagens = df.MATERIAL_GRUPO.value_counts().reindex(ordem)
    tons = (df.groupby("MATERIAL_GRUPO").PESO_LIQUIDO.sum() / 1000).reindex(ordem)
    tabela(pd.DataFrame({
        "pesagens": viagens,
        "pct_pesagens": (viagens / viagens.sum() * 100).round(2),
        "toneladas": tons.round(0),
        "pct_toneladas": (tons / tons.sum() * 100).round(2),
    }), "s1_classes")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cores = [CORES[0]] * 6 + [CINZA] * 2
    for ax, serie, titulo in [(axes[0], viagens, "Número de pesagens"),
                              (axes[1], tons, "Toneladas")]:
        y = np.arange(len(ordem))[::-1]
        ax.barh(y, serie.values, color=cores, height=0.65)
        total = serie.sum()
        for yi, v in zip(y, serie.values):
            ax.text(v + total * 0.005, yi, f"{num_br(v / total * 100)}%",
                    va="center", fontsize=8.5, color=TEXTO_2)
        ax.set_yticks(y, ordem)
        ax.set_title(titulo, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        eixo_milhar(ax, "x")
        ax.set_xlim(0, serie.max() * 1.18)
    fig.suptitle("Tipos de resíduo (variável-alvo)", x=0.01, y=0.995, ha="left",
                 fontweight="bold", fontsize=12)
    fig.text(0.01, 0.905, "Em azul, as 6 classes usadas no modelo. Em cinza, "
             "grupos que ficam de fora.", fontsize=9, color=TEXTO_2)
    fonte(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    salvar(fig, "04_classes_material")


def grafico_boxplot_classes(df):
    print("\n[6] Boxplot do peso líquido por classe")
    d = df[df.MATERIAL_GRUPO.isin(CLASSES_ALVO)].copy()
    d["t"] = d.PESO_LIQUIDO / 1000
    ordem = d.groupby("MATERIAL_GRUPO").t.median().sort_values().index

    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.boxplot(data=d, y="MATERIAL_GRUPO", x="t", order=ordem, ax=ax,
                color="#9ec5f4", width=0.55, linewidth=1,
                flierprops=dict(marker="o", markersize=2, alpha=0.25,
                                markerfacecolor=CINZA, markeredgecolor="none"),
                medianprops=dict(color=TEXTO, linewidth=1.6))
    ax.set_xlabel("Peso líquido por pesagem (toneladas)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.set_title("Cada tipo de resíduo tem um tamanho de carga típico", pad=22)
    subtitulo(ax, "A linha preta é a mediana. Os pontos são cargas fora do padrão "
                  "do próprio grupo.")
    fonte(fig)
    salvar(fig, "05_boxplot_peso_por_classe")

    resumo = d.groupby("MATERIAL_GRUPO").t.describe(percentiles=[.25, .5, .75]).round(2)
    tabela(resumo, "s1_peso_por_classe")


def grafico_dispersao(df):
    print("\n[7] Dispersão tara x peso líquido")
    d = df[df.CHASSI.isin(["TOCO", "TRUCK"])].sample(25_000, random_state=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (tipo, nome) in enumerate([("TOCO", "Toco (2 eixos)"),
                                      ("TRUCK", "Truck (3 eixos)")]):
        g = d[d.CHASSI == tipo]
        ax.scatter(g.TARA / 1000, g.PESO_LIQUIDO / 1000, s=6, alpha=0.25,
                   color=CORES[i], label=nome, linewidths=0)
    ax.set_xlabel("Tara - peso do caminhão vazio (t)")
    ax.set_ylabel("Peso líquido - carga (t)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 45)
    leg = ax.legend(markerscale=3, loc="upper right")
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.set_title("Caminhões mais pesados levam cargas maiores", pad=22)
    subtitulo(ax, "Amostra de 25 mil pesagens. Cargas acima de 20 t são quase "
                  "todas de entulho (61%) ou transbordo (31%).")
    fonte(fig)
    salvar(fig, "06_dispersao_tara_carga")


def grafico_correlacao(df):
    print("\n[8] Matriz de correlação")
    cols = {"PESO_BRUTO": "Peso bruto", "TARA": "Tara", "PESO_LIQUIDO": "Peso líquido",
            "CARGA_POR_TARA": "Carga ÷ tara", "ANO": "Ano", "MES": "Mês",
            "DIA_MES": "Dia do mês", "DIA_SEMANA": "Dia da semana",
            "FIM_DE_SEMANA": "Fim de semana", "AUTOMATICO": "Pesagem automática"}
    d = df.assign(AUTOMATICO=(df.TIPOCAD == "Automatico").astype(int))
    corr = d[list(cols)].corr().rename(index=cols, columns=cols)
    tabela(corr.round(3), "s1_correlacao")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mascara, cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=1.5,
                linecolor="#fcfcfb", cbar_kws={"shrink": 0.7}, ax=ax)
    for t in ax.texts:
        t.set_text(t.get_text().replace(".", ","))
    ax.grid(False)
    ax.set_title("Correlação entre os atributos numéricos (Pearson)", pad=22)
    subtitulo(ax, "Azul = sobem juntos; vermelho = um sobe e o outro desce; "
                  "cinza = sem relação.")
    fonte(fig, y=-0.1)
    salvar(fig, "07_matriz_correlacao")


def grafico_serie_diaria(dfp):
    print("\n[9] Toneladas por dia ao longo do tempo")
    d = dfp.groupby(["DATA", "ORIGEM"]).PESO_LIQUIDO.sum().unstack() / 1000
    media = d.rolling(28, center=True, min_periods=14).mean()

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for origem, cor in COR_ORIGEM.items():
        ax.plot(d.index, d[origem], color=cor, alpha=0.15, linewidth=0.8)
        ax.plot(media.index, media[origem], color=cor, label=origem)
    eixo_milhar(ax)
    ax.set_ylim(0)
    ax.set_xlim(d.index.min(), d.index.max())
    ax.set_ylim(0, 6800)
    eixo_datas(ax)
    ax.set_ylabel("Toneladas por dia")
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.45, 1.0))
    ax.set_title("Toneladas que chegam ao aterro por dia", pad=22)
    subtitulo(ax, "Linha clara = cada dia; linha forte = média de 4 semanas. "
                  "Transbordo fica de fora para não contar o mesmo lixo duas vezes.")
    fonte(fig)
    salvar(fig, "08_serie_toneladas_dia")


def grafico_dia_semana(dfp):
    print("\n[10] Toneladas por dia da semana")
    dia = dfp.groupby(["DATA"]).PESO_LIQUIDO.sum().div(1000).to_frame("t")
    dia["dow"] = dia.index.dayofweek
    dia["ano"] = dia.index.year

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.boxplot(data=dia, x="dow", y="t", hue="ano", ax=ax, width=0.7,
                palette=CORES[:3], linewidth=0.9, fliersize=2, gap=0.12)
    ax.set_xticks(range(7), DIAS)
    ax.set_xlabel("")
    ax.set_ylabel("Toneladas no dia")
    eixo_milhar(ax)
    ax.legend(title="", ncols=3, loc="upper right")
    ax.set_title("Domingo quase não tem coleta; terça é o dia mais pesado", pad=22)
    subtitulo(ax, "Cada caixa resume ~52 dias de um ano. Sem transbordo.")
    fonte(fig)
    salvar(fig, "09_boxplot_dia_semana")
    tabela(dia.groupby(["ano", "dow"]).t.median().unstack().round(0), "s1_dia_semana_mediana")


def grafico_mes_ano(dfp):
    print("\n[11] Mapa de calor mês x ano")
    d = dfp.groupby(["ANO", "MES"]).agg(t=("PESO_LIQUIDO", "sum"),
                                         dias=("DATA", "nunique"))
    tab = (d.t / 1000 / d.dias).unstack()
    tab.columns = MESES

    fig, ax = plt.subplots(figsize=(10, 2.8))
    sns.heatmap(tab, cmap=MAPA_SEQUENCIAL, annot=True, fmt=".0f",
                annot_kws={"size": 8.5}, linewidths=2, linecolor="#fcfcfb",
                cbar_kws={"label": "t/dia"}, ax=ax)
    for t in ax.texts:
        t.set_text(num_br(float(t.get_text()), 0))
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.grid(False)
    ax.set_title("Média de toneladas por dia, em cada mês", pad=22)
    subtitulo(ax, "Sem transbordo. 2024 fica bem abaixo: parte das pesagens pode "
                  "faltar no arquivo (ver texto).")
    fonte(fig)
    salvar(fig, "10_mapa_calor_mes_ano")
    tabela(tab.round(0), "s1_toneladas_dia_mes_ano")


def grafico_regionais(dfp):
    print("\n[12] Toneladas por regional")
    d = dfp[dfp.REGIONAL.str.startswith("SER")]
    dias = dfp.DATA.nunique()
    t = (d.groupby("REGIONAL").PESO_LIQUIDO.sum() / 1000 / dias).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(t.index, t.values, color=CORES[0], height=0.65)
    for i, v in enumerate(t.values):
        ax.text(v + 4, i, num_br(v, 0), va="center", fontsize=8.5, color=TEXTO_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Toneladas por dia (média 2022-2024)")
    ax.set_title("Toneladas por Secretaria Regional (SER) de Fortaleza", pad=22)
    subtitulo(ax, "Só a coleta ligada a uma regional. Caucaia e grandes geradores "
                  "ficam de fora.")
    fonte(fig)
    salvar(fig, "11_toneladas_por_regional")


def grafico_material_por_regional(df):
    print("\n[13] Composição do resíduo em cada regional")
    d = df[df.MATERIAL_GRUPO.isin(CLASSES_ALVO)].copy()
    d["LOCAL"] = np.where(d.REGIONAL.str.startswith("SER"), d.REGIONAL, d.ORIGEM)
    ordem_ser = ["SER " + r for r in
                 ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]]
    linhas = ordem_ser + ["Caucaia", "Grandes geradores e outros"]
    tab = pd.crosstab(d.LOCAL, d.MATERIAL_GRUPO, values=d.PESO_LIQUIDO,
                      aggfunc="sum", normalize="index").reindex(linhas)[CLASSES_ALVO] * 100
    tab.index = [i.replace("Grandes geradores e outros", "Grandes geradores") for i in tab.index]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.heatmap(tab, cmap=MAPA_SEQUENCIAL, annot=True, fmt=".0f", vmin=0, vmax=100,
                annot_kws={"size": 8.5}, linewidths=1.5, linecolor="#fcfcfb",
                cbar_kws={"label": "% das toneladas da linha"}, ax=ax)
    ax.set_xticklabels([ROTULO_CURTO[c] for c in CLASSES_ALVO], rotation=0, fontsize=8.5)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(False)
    ax.set_title("Do que é feito o resíduo de cada lugar (% do peso)", pad=22)
    subtitulo(ax, "Cada linha soma 100%.")
    fonte(fig)
    salvar(fig, "12_material_por_regional")
    tabela(tab.round(1), "s1_material_por_local")


def nome_curto(nome):
    for fim in [" LTDA - EPP", " LTDA - ME", " LTDA", " S/A", " SA", " EIRELI"]:
        nome = nome.removesuffix(fim)
    return nome


def grafico_transportadoras(dfp):
    print("\n[14] Transportadoras")
    t = dfp.groupby("TRANSPORTADORA").PESO_LIQUIDO.sum().sort_values(ascending=False) / 1e6
    top = t.head(10)[::-1]          
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh([nome_curto(n) for n in top.index], top.values, color=CORES[0], height=0.65)
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i,
                f"{num_br(v, 0)} mil t  ({num_br(v / t.sum() * 100)}%)",
                va="center", fontsize=8.5, color=TEXTO_2)
    ax.set_xlim(0, top.max() * 1.3)
    eixo_milhar(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mil toneladas (2022-2024)")
    ax.set_title("As 10 empresas que mais levaram resíduo ao aterro", pad=22)
    subtitulo(ax, f"São {t.size} transportadoras no total; a primeira sozinha "
                  f"leva {num_br(t.iloc[0] / t.sum() * 100, 0)}% do peso.")
    fonte(fig)
    salvar(fig, "13_top_transportadoras")
    tabela((t * 1000).round(0).to_frame("toneladas"), "s1_transportadoras")


def grafico_classes_por_ano(df):
    print("\n[15] Classes por ano")
    d = df[df.MATERIAL_GRUPO.isin(CLASSES_ALVO)]
    tab = pd.crosstab(d.MATERIAL_GRUPO, d.ANO, normalize="columns").reindex(CLASSES_ALVO) * 100

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(CLASSES_ALVO))
    w = 0.26
    for i, ano in enumerate(tab.columns):
        ax.bar(x + (i - 1) * w, tab[ano], width=w - 0.03, color=CORES[i], label=str(ano))
    ax.set_xticks(x, [ROTULO_CURTO[c] for c in CLASSES_ALVO], fontsize=8.5)
    ax.set_ylabel("% das pesagens do ano")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=3)
    ax.set_title("2022 e 2023 são parecidos; 2024 tem mais domiciliar", pad=22)
    subtitulo(ax, "e menos lixo especial urbano. A ordem das classes, porém, "
                  "é a mesma nos três anos.")
    fonte(fig)
    salvar(fig, "14_classes_por_ano")


def main():
    aplicar_estilo()
    print("Lendo os 3 arquivos...")
    brutos = padronizar_textos(carregar_brutos())
    descrever_base(brutos)

    limpos = criar_atributos(remover_pesos_invalidos(remover_duplicatas(brutos)))

    sem_transbordo = limpos[limpos.MATERIAL_GRUPO != "Transbordo"]

    grafico_registros_por_mes(brutos, limpos)
    grafico_inativos(brutos)
    grafico_histogramas(limpos)
    grafico_classes(limpos)
    grafico_boxplot_classes(limpos)
    grafico_dispersao(limpos)
    grafico_correlacao(limpos)
    grafico_serie_diaria(sem_transbordo)
    grafico_dia_semana(sem_transbordo)
    grafico_mes_ano(sem_transbordo)
    grafico_regionais(sem_transbordo)
    grafico_material_por_regional(limpos)
    grafico_transportadoras(sem_transbordo)
    grafico_classes_por_ano(limpos)
    print("\nPronto.")


if __name__ == "__main__":
    main()
