"""Regressão: estimar o peso líquido (t) de cada pesagem.

Usa as mesmas linhas de treino e teste da classificação (Sprint 2) e o
mesmo protocolo: busca de hiperparâmetros por validação cruzada de 5 folds
numa amostra de 80.000 linhas do treino, ajuste final no treino inteiro e
uma única avaliação no teste.

Rode a partir de src/, depois do pré-processamento (e da Sprint 4, para
a comparação entre as tarefas):

    python sprint2_preprocessamento.py
    python sprint5_regressao.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from config import CINZA, CLASSES_ALVO, CORES, DADOS_PROCESSADOS, SEMENTE, TABELAS, TEXTO_2
from graficos import aplicar_estilo, fonte, num_br, salvar

N_BUSCA = 80_000
FOLDS = KFold(5, shuffle=True, random_state=SEMENTE)


def carregar():
    tr = pd.read_csv(DADOS_PROCESSADOS / "regressao_treino.csv.gz")
    te = pd.read_csv(DADOS_PROCESSADOS / "regressao_teste.csv.gz")
    X_tr, y_tr = tr.drop(columns=["ALVO", "ANO"]).astype("float32"), tr["ALVO"].to_numpy()
    X_te, y_te = te.drop(columns=["ALVO", "ANO"]).astype("float32"), te["ALVO"].to_numpy()
    return X_tr, y_tr, X_te, y_te, te["ANO"].to_numpy()


def metricas(y, p):
    erro = y - p
    return {"MAE": np.abs(erro).mean(), "RMSE": np.sqrt((erro ** 2).mean()),
            "R2": 1 - (erro ** 2).sum() / ((y - y.mean()) ** 2).sum()}


def classe_de(X):
    cols = [c for c in X.columns if c.startswith("MATERIAL_GRUPO_")]
    return X[cols].to_numpy().argmax(1), [c.removeprefix("MATERIAL_GRUPO_") for c in cols]


def main():
    aplicar_estilo()
    X_tr, y_tr, X_te, y_te, ano_te = carregar()
    rng = np.random.default_rng(SEMENTE)
    idx = rng.choice(len(X_tr), N_BUSCA, replace=False)
    Xb, yb = X_tr.iloc[idx], y_tr[idx]

    modelos = {
        "Regressão linear (Ridge)": (Ridge(), {"alpha": [0.1, 1, 10]}),
        "Árvore de decisão": (DecisionTreeRegressor(min_samples_leaf=50, random_state=SEMENTE),
                              {"max_depth": [12, 24, None]}),
        "Rede neural": (MLPRegressor(learning_rate_init=0.001, batch_size=512, alpha=1e-4,
                                     max_iter=25, early_stopping=True, validation_fraction=0.1,
                                     random_state=SEMENTE),
                        {"hidden_layer_sizes": [(64,), (128,)]}),
        "Random Forest": (RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=SEMENTE),
                          {"max_features": ["sqrt", 0.3], "min_samples_leaf": [10, 30]}),
        "Gradient Boosting": (HistGradientBoostingRegressor(max_iter=400, early_stopping=True,
                                                            validation_fraction=0.1,
                                                            n_iter_no_change=10, random_state=SEMENTE),
                              {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 127]}),
    }
    familia = {"Regressão linear (Ridge)": "Individual", "Árvore de decisão": "Individual",
               "Rede neural": "Individual", "Random Forest": "Homogêneo",
               "Gradient Boosting": "Homogêneo"}

    linhas, previsoes, busca = [], {}, []
    # referências
    media = y_tr.mean()
    k_tr, nomes = classe_de(X_tr)
    k_te, _ = classe_de(X_te)
    media_classe = pd.Series(y_tr).groupby(k_tr).mean()
    for nome, p in [("Referência: média geral", np.full_like(y_te, media)),
                    ("Referência: média do tipo de resíduo", media_classe.reindex(k_te).to_numpy())]:
        linhas.append({"modelo": nome, "familia": "Referência", **metricas(y_te, p),
                       "MAE_cv": np.nan, "treino_s": 0.0})
        previsoes[nome] = p

    for nome, (est, grade) in modelos.items():
        print(f"\n[{nome}] busca em {N_BUSCA} linhas, 5 folds")
        t0 = time.time()
        gs = GridSearchCV(est, grade, cv=FOLDS, scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(Xb, yb)
        for params, m, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"],
                                gs.cv_results_["std_test_score"]):
            busca.append({"modelo": nome, "parametros": str(params), "MAE_cv": -m, "desvio": s})
        print(f"  melhor: {gs.best_params_}  MAE CV = {-gs.best_score_:.3f}  ({time.time() - t0:.0f}s)")
        final = est.set_params(**gs.best_params_)
        if nome == "Random Forest":
            final.set_params(n_estimators=100)
        t0 = time.time()
        final.fit(X_tr, y_tr)
        t_treino = time.time() - t0
        p = final.predict(X_te)
        previsoes[nome] = p
        linhas.append({"modelo": nome, "familia": familia[nome], **metricas(y_te, p),
                       "MAE_cv": -gs.best_score_, "treino_s": t_treino,
                       "escolhido": str(gs.best_params_)})
        print(f"  teste: MAE {linhas[-1]['MAE']:.3f}  R2 {linhas[-1]['R2']:.3f}  treino {t_treino:.0f}s")

    # comitê heterogêneo: média das previsões de árvore, rede e Gradient Boosting
    trio = ["Árvore de decisão", "Rede neural", "Gradient Boosting"]
    p = np.mean([previsoes[m] for m in trio], axis=0)
    previsoes["Média (árvore + rede + GB)"] = p
    linhas.append({"modelo": "Média (árvore + rede + GB)", "familia": "Heterogêneo",
                   **metricas(y_te, p), "MAE_cv": np.nan,
                   "treino_s": sum(r["treino_s"] for r in linhas if r["modelo"] in trio)})

    res = pd.DataFrame(linhas)
    res.round(4).to_csv(TABELAS / "s5_regressao_resultados.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(busca).round(4).to_csv(TABELAS / "s5_regressao_busca.csv", index=False,
                                        encoding="utf-8-sig")
    print(res.round(3).to_string())

    melhor = res[res.familia != "Referência"].sort_values("MAE").iloc[0]["modelo"]
    pm = previsoes[melhor]
    # erro por tipo de resíduo e por ano
    por_classe = pd.DataFrame({"classe": [nomes[k] for k in k_te], "erro": np.abs(y_te - pm),
                               "y": y_te}).groupby("classe").agg(
        MAE=("erro", "mean"), carga_media=("y", "mean"), n=("y", "size"))
    por_classe["MAE_relativo"] = por_classe.MAE / por_classe.carga_media
    por_classe = por_classe.reindex(CLASSES_ALVO)
    por_classe.round(3).to_csv(TABELAS / "s5_regressao_por_classe.csv", encoding="utf-8-sig")
    por_ano = pd.DataFrame([{"ano": a, "n": int((ano_te == a).sum()),
                             **metricas(y_te[ano_te == a], pm[ano_te == a])} for a in (2022, 2023, 2024)])
    por_ano.round(3).to_csv(TABELAS / "s5_regressao_por_ano.csv", index=False, encoding="utf-8-sig")
    print(melhor); print(por_classe.round(3)); print(por_ano.round(3))

    grafico_mae(res)
    grafico_real_previsto(y_te, pm, melhor)
    comparar_tarefas(res, melhor)


def comparar_tarefas(res, melhor):
    """Quanto cada tarefa reduziu o erro em relação à sua referência."""
    arq = TABELAS / "s4_metricas.csv"
    if not arq.exists():
        print("  (sem s4_metricas.csv: rode a Sprint 4 para comparar as tarefas)")
        return
    s4 = pd.read_csv(arq)
    s4 = s4[s4.familia != "Referência"] if "familia" in s4 else s4
    c = s4.sort_values("f1_macro", ascending=False).iloc[0]
    y_te_cls = pd.read_csv(DADOS_PROCESSADOS / "teste.csv.gz", usecols=["ALVO"]).ALVO
    y_tr_cls = pd.read_csv(DADOS_PROCESSADOS / "treino.csv.gz", usecols=["ALVO"]).ALVO
    erro_ref_cls = 1 - (y_te_cls == y_tr_cls.mode()[0]).mean()
    erro_cls = 1 - c["acuracia"]
    ref = res.set_index("modelo").loc["Referência: média geral", "MAE"]
    mae = res.set_index("modelo").loc[melhor, "MAE"]
    tab = pd.DataFrame([
        {"tarefa": "Classificação", "medida": "Taxa de erro (1 - acurácia)",
         "melhor_modelo": c["modelo"], "erro_referencia": erro_ref_cls, "erro_melhor": erro_cls,
         "reducao": 1 - erro_cls / erro_ref_cls},
        {"tarefa": "Regressão", "medida": "MAE (t)", "melhor_modelo": melhor,
         "erro_referencia": ref, "erro_melhor": mae, "reducao": 1 - mae / ref},
    ]).round(4)
    tab.to_csv(TABELAS / "s5_comparacao_tarefas.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string())


def grafico_mae(res):
    d = res.sort_values("MAE", ascending=False)
    cor = {"Referência": CINZA, "Individual": CORES[0], "Homogêneo": CORES[1], "Heterogêneo": CORES[2]}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(d))
    ax.barh(y, d.MAE, color=[cor[f] for f in d.familia], height=0.62)
    for yi, (m, r2) in enumerate(zip(d.MAE, d.R2)):
        ax.text(m + 0.04, yi, f"{num_br(m, 2)} t  (R² {num_br(abs(r2) if abs(r2) < 0.005 else r2, 2)})", va="center",
                fontsize=8.5, color=TEXTO_2)
    ax.set_yticks(y, d.modelo)
    ax.set_xlim(0, d.MAE.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Erro médio absoluto no teste (toneladas) — menor é melhor")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in cor.items()],
              loc="upper right", fontsize=8.5)
    ax.set_title("Erro de cada modelo ao estimar o peso da carga", pad=12)
    fonte(fig)
    salvar(fig, "01_regressao_mae", "sprint5")


def grafico_real_previsto(y, p, nome):
    rng = np.random.default_rng(1)
    i = rng.choice(len(y), 20_000, replace=False)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(y[i], p[i], s=4, alpha=0.2, color=CORES[0], linewidths=0)
    lim = [0, 45]
    ax.plot(lim, lim, color=TEXTO_2, linewidth=1, linestyle="--")
    ax.text(38, 40.5, "acerto perfeito", fontsize=8.5, color=TEXTO_2, ha="right")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Peso líquido real (t)")
    ax.set_ylabel("Peso líquido previsto (t)")
    ax.set_title(f"{nome}: previsto contra real", pad=12)
    fonte(fig, "Amostra de 20 mil pesagens do teste.")
    salvar(fig, "02_regressao_real_previsto", "sprint5")


if __name__ == "__main__":
    main()
