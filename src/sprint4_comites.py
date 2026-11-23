"""Sprint 4: comitês homogêneos e heterogêneos.

Mantém o protocolo da Sprint 3: a divisão 70/30 da Sprint 2 fica intocada,
os hiperparâmetros são escolhidos por validação cruzada estratificada
(5 folds) numa amostra de 80.000 linhas do treino, o ajuste final usa o
treino inteiro e o teste só entra na avaliação final.

Rode a partir de src/, depois das sprints anteriores:

    python sprint2_preprocessamento.py
    python sprint3_modelos.py
    python sprint4_comites.py

O empilhamento com Random Forest e Gradient Boosting é a parte mais cara.
Para uma rodada rápida, mude EMPILHAR_COMITES para False.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config import CINZA, CLASSES_ALVO, CORES, ROTULO_CURTO, SEMENTE, TABELAS, TEXTO_2
from graficos import MAPA_SEQUENCIAL, aplicar_estilo, fonte, num_br, salvar, subtitulo
from sprint3_modelos import (
    AMOSTRA_BUSCA,
    N_FOLDS,
    carregar_parte,
    indices_estratificados,
    metricas_globais,
    metricas_por_classe,
    resumir_params,
)

PASTA_FIG = "sprint4"
FOLDS_EMPILHAMENTO = 5
EMPILHAR_COMITES = True

COR_FAMILIA = {"Individual": CORES[0], "Homogêneo": CORES[1], "Heterogêneo": CORES[2]}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The line search", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ============================================================ configurações
def rede_neural():
    """Mesma rede escolhida na Sprint 3."""
    return MLPClassifier(
        hidden_layer_sizes=(128,), activation="relu", solver="adam",
        learning_rate_init=1e-3, batch_size=512, alpha=1e-4, max_iter=25,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=4,
        random_state=SEMENTE)


def arvore():
    """Mesma árvore escolhida na Sprint 3."""
    return DecisionTreeClassifier(
        criterion="gini", max_depth=24, min_samples_leaf=50,
        class_weight="balanced", random_state=SEMENTE)


def individuais():
    """Os cinco algoritmos da Sprint 3 avaliados no teste inteiro, com os
    hiperparâmetros que venceram a busca. O KNN fica de fora porque usou
    protocolo reduzido e não tem previsão para o teste inteiro."""
    return {
        "Rede neural": {"modelo": rede_neural(), "peso": True},
        "Árvore de decisão": {"modelo": arvore(), "peso": False},
        "Regressão logística": {"modelo": LogisticRegression(
            C=10.0, solver="lbfgs", max_iter=250, tol=1e-3,
            class_weight="balanced", random_state=SEMENTE), "peso": False},
        "SVM linear": {"modelo": LinearSVC(
            C=1.0, class_weight="balanced", dual="auto", max_iter=2000,
            random_state=SEMENTE), "peso": False},
        "Naive Bayes": {"modelo": GaussianNB(var_smoothing=1e-9), "peso": False},
    }


def homogeneos():
    """Comitês do mesmo algoritmo-base. Quem não tem grade herda os
    hiperparâmetros do algoritmo individual da Sprint 3."""
    return [
        {
            "nome": "Bagging de árvores",
            "base": "Árvore de decisão",
            "metodo": "Bagging: cada árvore treina numa amostra bootstrap; média das probabilidades",
            "estimator": BaggingClassifier(
                estimator=arvore(), n_estimators=50, max_samples=1.0,
                bootstrap=True, n_jobs=-1, random_state=SEMENTE),
            "param_grid": {},
            "fixos": {"n_estimators": 50, "max_samples": 1.0, "bootstrap": True,
                      "árvore": "gini, max_depth=24, min_samples_leaf=50, balanced"},
            "peso": False,
        },
        {
            "nome": "Random Forest",
            "base": "Árvore de decisão",
            "metodo": "Bagging + sorteio de atributos em cada divisão; média das probabilidades",
            "estimator": RandomForestClassifier(
                n_estimators=100, criterion="gini",
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=SEMENTE),
            # Folhas com no mínimo 10 pesagens: árvores sem limite e com folha 1
            # passaram de 8 GB de memória no treino inteiro.
            "param_grid": {"max_features": ["sqrt", 0.3],
                           "min_samples_leaf": [10, 30]},
            "fixos": {"n_estimators": 100, "criterion": "gini",
                      "class_weight": "balanced_subsample"},
            "peso": False,
        },
        {
            "nome": "AdaBoost",
            "base": "Árvore de decisão",
            "metodo": "Boosting: cada árvore dá mais peso aos erros da anterior; voto ponderado",
            "estimator": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=SEMENTE),
                n_estimators=100, random_state=SEMENTE),
            "param_grid": {"estimator__max_depth": [4, 8],
                           "learning_rate": [0.5, 1.0]},
            "fixos": {"n_estimators": 100, "algoritmo": "SAMME",
                      "peso_inicial": "balanceado por classe"},
            "peso": True,
        },
        {
            "nome": "Gradient Boosting",
            "base": "Árvore de decisão",
            "metodo": "Boosting: cada árvore corrige o resíduo das anteriores; soma das saídas",
            "estimator": HistGradientBoostingClassifier(
                max_iter=400, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=10, class_weight="balanced",
                random_state=SEMENTE),
            "param_grid": {"learning_rate": [0.05, 0.1],
                           "max_leaf_nodes": [31, 127]},
            "fixos": {"max_iter": 400, "early_stopping": True,
                      "n_iter_no_change": 10, "class_weight": "balanced"},
            "peso": False,
        },
        {
            "nome": "Bagging de redes neurais",
            "base": "Rede neural",
            "metodo": "Bagging: cada rede treina numa amostra bootstrap; média das probabilidades",
            "estimator": BaggingClassifier(
                estimator=rede_neural(), n_estimators=5, max_samples=1.0,
                bootstrap=True, n_jobs=1, random_state=SEMENTE),
            "param_grid": {},
            "fixos": {"n_estimators": 5, "max_samples": 1.0, "bootstrap": True,
                      "rede": "128 neurônios, ReLU, Adam, até 25 épocas"},
            "peso": True,
        },
    ]


TRES_MELHORES = ["Rede neural", "Árvore de decisão", "Regressão logística"]
CINCO = list(individuais())


def heterogeneos():
    """Combinações de algoritmos diferentes. Votação usa os modelos já
    ajustados; empilhamento treina um meta-modelo nas previsões fora do fold."""
    lista = [
        {"nome": "Votação majoritária (5 algoritmos)", "tipo": "voto",
         "bases": CINCO},
        {"nome": "Votação majoritária (3 melhores)", "tipo": "voto",
         "bases": TRES_MELHORES},
        {"nome": "Votação suave (3 melhores)", "tipo": "suave",
         "bases": TRES_MELHORES},
        {"nome": "Stacking (5 algoritmos)", "tipo": "stacking", "bases": CINCO},
        {"nome": "Stacking (árvore + rede)", "tipo": "stacking",
         "bases": ["Rede neural", "Árvore de decisão"]},
    ]
    if EMPILHAR_COMITES:
        lista.append({"nome": "Stacking (RF + Gradient Boosting + rede)",
                      "tipo": "stacking",
                      "bases": ["Random Forest", "Gradient Boosting", "Rede neural"]})
    return lista


def meta_modelo():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                           class_weight="balanced", random_state=SEMENTE))


# ================================================================ utilidades
def pesos(y):
    return compute_sample_weight("balanced", y)


def treinar(modelo, X, y, usa_peso):
    params = {"sample_weight": pesos(y)} if usa_peso else {}
    t0 = time.perf_counter()
    modelo.fit(X, y, **params)
    return modelo, time.perf_counter() - t0


def pontuar(modelo, X):
    """Probabilidades quando o modelo tem; senão, a função de decisão (SVM)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    return modelo.decision_function(X)


def prever(modelo, X):
    t0 = time.perf_counter()
    pred = modelo.predict(X)
    return pred, time.perf_counter() - t0


def buscar(spec, X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    busca = GridSearchCV(spec["estimator"], spec["param_grid"], scoring="f1_macro",
                         cv=cv, n_jobs=-1, refit=False, error_score="raise")
    params = {"sample_weight": pesos(y)} if spec["peso"] else {}
    t0 = time.perf_counter()
    busca.fit(X, y, **params)
    segundos = time.perf_counter() - t0
    tabela = pd.DataFrame({
        "modelo": spec["nome"],
        "params": [resumir_params(p) for p in busca.cv_results_["params"]],
        "f1_macro_cv": busca.cv_results_["mean_test_score"],
        "f1_macro_cv_std": busca.cv_results_["std_test_score"],
        "rank": busca.cv_results_["rank_test_score"],
        "folds": N_FOLDS,
        "linhas_busca": len(y),
    })
    melhor = tabela.loc[tabela["rank"] == 1].iloc[0]
    return busca.best_params_, melhor, segundos, tabela


def votar(predicoes, ordem):
    """Votação majoritária. Empate vai para o modelo que ficou melhor na
    validação da Sprint 3 (a ordem da lista)."""
    votos = np.zeros((len(predicoes[0]), len(CLASSES_ALVO)))
    indice = {c: i for i, c in enumerate(CLASSES_ALVO)}
    for k, pred in enumerate(predicoes):
        bonus = 1 + 1e-3 * (len(ordem) - k)
        cols = np.fromiter((indice[p] for p in pred), dtype=int, count=len(pred))
        votos[np.arange(len(pred)), cols] += bonus
    return np.asarray(CLASSES_ALVO, dtype=object)[votos.argmax(axis=1)]


def suave(probas, classes):
    media = np.mean(probas, axis=0)
    return np.asarray(classes, dtype=object)[media.argmax(axis=1)]


def curva_n_modelos(nome, modelo, X_te, y_te, passos):
    """F1 macro no teste usando só os k primeiros modelos do comitê.
    Serve só para análise: nenhum hiperparâmetro é escolhido aqui."""
    linhas = []
    X_np = np.asarray(X_te, dtype=np.float32)
    classes = np.asarray(modelo.classes_, dtype=object)
    if isinstance(modelo, (BaggingClassifier, RandomForestClassifier)):
        feats = getattr(modelo, "estimators_features_", None)
        soma = None
        for i, est in enumerate(modelo.estimators_):
            Xi = X_np if feats is None else X_np[:, feats[i]]
            p = est.predict_proba(Xi)
            soma = p if soma is None else soma + p
            if i + 1 in passos:
                pred = classes[soma.argmax(axis=1)]
                linhas.append({"comite": nome, "n_modelos": i + 1,
                               "f1_macro": f1_score(y_te, pred, average="macro")})
    else:  # AdaBoost e Gradient Boosting têm previsão estágio a estágio
        for i, pred in enumerate(modelo.staged_predict(X_te)):
            if i + 1 in passos:
                linhas.append({"comite": nome, "n_modelos": i + 1,
                               "f1_macro": f1_score(y_te, pred, average="macro")})
    return linhas


def diversidade(preds, y):
    nomes = list(preds)
    y = np.asarray(y, dtype=object)
    disc = pd.DataFrame(0.0, index=nomes, columns=nomes)
    duplo = pd.DataFrame(0.0, index=nomes, columns=nomes)
    for a in nomes:
        for b in nomes:
            pa, pb = np.asarray(preds[a], dtype=object), np.asarray(preds[b], dtype=object)
            disc.loc[a, b] = np.mean(pa != pb)
            duplo.loc[a, b] = np.mean((pa != y) & (pb != y))
    return disc, duplo


def discordancia_media(preds, nomes):
    pares = [np.mean(np.asarray(preds[a], dtype=object) != np.asarray(preds[b], dtype=object))
             for i, a in enumerate(nomes) for b in nomes[i + 1:]]
    return float(np.mean(pares)) if pares else np.nan


def oraculo(preds, nomes, y):
    """Fração em que pelo menos um dos modelos acerta: o teto da combinação."""
    y = np.asarray(y, dtype=object)
    acerto = np.zeros(len(y), dtype=bool)
    for n in nomes:
        acerto |= np.asarray(preds[n], dtype=object) == y
    return acerto.mean()


# ================================================================== gráficos
NOME_CURTO = {
    "Rede neural": "Rede neural",
    "Árvore de decisão": "Árvore",
    "Regressão logística": "Logística",
    "SVM linear": "SVM",
    "Naive Bayes": "Naive Bayes",
}


def legenda_familias():
    return [Patch(facecolor=cor, label=fam) for fam, cor in COR_FAMILIA.items()]


def grafico_geral(tabela):
    d = tabela.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y, d.f1_macro, color=[COR_FAMILIA[f] for f in d.familia], height=0.66)
    for yi, v in zip(y, d.f1_macro):
        ax.text(v + 0.006, yi, num_br(v * 100) + "%", va="center", fontsize=8,
                color=TEXTO_2)
    ax.set_yticks(y, d.modelo, fontsize=8.5)
    menor = max(0.0, np.floor((d.f1_macro.min() - 0.05) * 10) / 10)
    ax.set_xlim(menor, 1.02)
    ax.set_xlabel("F1 macro no teste")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=legenda_familias(), loc="lower right", fontsize=8)
    ax.set_title("F1 macro no teste: individuais, comitês homogêneos e heterogêneos",
                 pad=22)
    subtitulo(ax, f"O eixo começa em {num_br(menor * 100, 0)}% para separar os "
                  "modelos próximos. Teste fechado de 213 mil pesagens.")
    fonte(fig, "Fonte: experimentos da Sprint 4, conjunto de teste.")
    salvar(fig, "01_f1_macro_geral", PASTA_FIG)


def grafico_homogeneo_vs_base(tabela):
    hom = tabela[tabela.familia == "Homogêneo"]
    f1_ind = tabela[tabela.familia == "Individual"].set_index("modelo")["f1_macro"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(hom))
    base = [f1_ind[b] for b in hom.base]
    ax.bar(x - 0.19, base, width=0.36, color=CORES[0], label="Algoritmo individual")
    ax.bar(x + 0.19, hom.f1_macro, width=0.36, color=CORES[1], label="Comitê")
    for xi, b, c in zip(x, base, hom.f1_macro):
        delta = (c - b) * 100
        sinal = "+" if delta >= 0 else "−"
        ax.text(xi + 0.19, c + 0.004, f"{sinal}{num_br(abs(delta))} p.p.",
                ha="center", fontsize=8, color=TEXTO_2)
    rotulos = [f"{n}\n(base: {NOME_CURTO.get(b, b)})" for n, b in zip(hom.modelo, hom.base)]
    ax.set_xticks(x, rotulos, fontsize=8)
    baixo = max(0.0, np.floor((min(min(base), hom.f1_macro.min()) - 0.03) * 20) / 20)
    ax.set_ylim(baixo, 1.0)
    ax.set_ylabel("F1 macro no teste")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Cada comitê homogêneo contra o seu algoritmo-base", pad=22)
    subtitulo(ax, "A diferença em pontos percentuais está acima de cada barra laranja. "
                  f"Eixo a partir de {num_br(baixo * 100, 0)}%.")
    fonte(fig, "Fonte: experimentos da Sprint 4, conjunto de teste.")
    salvar(fig, "02_homogeneos_vs_individual", PASTA_FIG)


def grafico_curva(curva, f1_ind):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    arvores = ["Bagging de árvores", "Random Forest", "AdaBoost", "Gradient Boosting"]
    for i, nome in enumerate(arvores):
        d = curva[curva.comite == nome]
        if d.empty:
            continue
        ax.plot(d.n_modelos, d.f1_macro, color=CORES[i + 1], marker="o", markersize=4,
                label=nome)
    ax.axhline(f1_ind["Árvore de decisão"], color=CINZA, linestyle="--", linewidth=1.2,
               label="Árvore individual")
    ax.set_xscale("log")
    ax.set_xlabel("Número de árvores no comitê (escala log)")
    ax.set_ylabel("F1 macro no teste")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Comitês de árvores", fontsize=10.5)

    ax = axes[1]
    d = curva[curva.comite == "Bagging de redes neurais"]
    if not d.empty:
        ax.plot(d.n_modelos, d.f1_macro, color=CORES[1], marker="o", markersize=5,
                label="Bagging de redes")
        valores = list(d.f1_macro) + [f1_ind["Rede neural"]]
        folga = max(0.005, (max(valores) - min(valores)) * 0.4)
        ax.set_ylim(min(valores) - folga, max(valores) + folga)
    ax.axhline(f1_ind["Rede neural"], color=CINZA, linestyle="--", linewidth=1.2,
               label="Rede individual")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Número de redes no comitê")
    ax.set_title("Comitê de redes neurais", fontsize=10.5)
    fig.suptitle("O que acontece com o F1 macro quando o comitê cresce",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fonte(fig, "Fonte: Sprint 4. Curva só para análise: os hiperparâmetros vieram "
               "da validação, não do teste.")
    fig.tight_layout()
    salvar(fig, "03_curva_n_modelos", PASTA_FIG)


def grafico_diversidade(disc, duplo):
    nomes = [NOME_CURTO.get(n, n) for n in disc.index]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, mat, titulo in [
            (axes[0], disc, "Discordância (% das pesagens)"),
            (axes[1], duplo, "Erro duplo (os dois erram, %)")]:
        v = mat.to_numpy() * 100
        im = ax.imshow(v, cmap=MAPA_SEQUENCIAL, vmin=0, vmax=max(v.max(), 1))
        for i in range(len(nomes)):
            for j in range(len(nomes)):
                cor = "white" if v[i, j] > 0.6 * v.max() else TEXTO_2
                ax.text(j, i, num_br(v[i, j]), ha="center", va="center", fontsize=8,
                        color=cor)
        ax.set_xticks(range(len(nomes)), nomes, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(nomes)), nomes, fontsize=8)
        ax.set_title(titulo, fontsize=10.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Diversidade entre os algoritmos individuais",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fonte(fig, "Fonte: Sprint 4, previsões no teste. Diagonal da direita = taxa de erro "
               "do próprio modelo.")
    fig.tight_layout()
    salvar(fig, "04_diversidade", PASTA_FIG)


def grafico_custo(tabela):
    d = tabela.sort_values("tempo_total_s")
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y, d.tempo_total_s, color=[COR_FAMILIA[f] for f in d.familia], height=0.66)
    for yi, t, f1 in zip(y, d.tempo_total_s, d.f1_macro):
        ax.text(t * 1.12, yi, f"{num_br(t, 0 if t >= 10 else 1)} s · F1 {num_br(f1 * 100)}%",
                va="center", fontsize=8, color=TEXTO_2)
    ax.set_yticks(y, d.modelo, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(d.tempo_total_s.min() * 0.5, d.tempo_total_s.max() * 12)
    ax.set_xlabel("Tempo total de treino em segundos (escala log)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=legenda_familias(), fontsize=8, loc="lower right")
    ax.set_title("Custo de treino de cada modelo, com o F1 macro ao lado", pad=22)
    subtitulo(ax, "Votação e stacking somam o tempo dos modelos que combinam.")
    fonte(fig, "Fonte: Sprint 4. Tempos medidos na mesma máquina, numa única execução.")
    salvar(fig, "05_custo_vs_f1", PASTA_FIG)


def grafico_por_classe(por_classe, modelos):
    d = por_classe[por_classe.modelo.isin(modelos)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(CLASSES_ALVO))
    largura = 0.8 / len(modelos)
    for i, nome in enumerate(modelos):
        vals = d[d.modelo == nome].set_index("classe").reindex(CLASSES_ALVO)["f1"]
        ax.bar(x + (i - (len(modelos) - 1) / 2) * largura, vals, width=largura * 0.92,
               color=CORES[i], label=nome)
    ax.set_xticks(x, [ROTULO_CURTO[c] for c in CLASSES_ALVO], fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("F1")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_title("F1 de cada classe: o melhor modelo de cada família", pad=22)
    subtitulo(ax, "Individual em azul, comitê homogêneo em laranja, heterogêneo em verde.")
    fonte(fig, "Fonte: experimentos da Sprint 4, conjunto de teste.")
    salvar(fig, "06_f1_por_classe", PASTA_FIG)


def grafico_confusao(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES_ALVO)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    rotulos = [ROTULO_CURTO[c].replace("\n", " ") for c in CLASSES_ALVO]
    ax.set_xticks(range(6), rotulos, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(6), rotulos, fontsize=8)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    for i in range(6):
        for j in range(6):
            cor = "white" if pct[i, j] > 0.55 else TEXTO_2
            ax.text(j, i, num_br(pct[i, j] * 100, 0) + "%", ha="center", va="center",
                    fontsize=8, color=cor)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(titulo, pad=14)
    ax.grid(False)
    fonte(fig, "Fonte: Sprint 4. Cada linha soma 100% da classe verdadeira.", y=-0.08)
    salvar(fig, "07_matriz_confusao", PASTA_FIG)
    return cm


# ====================================================================== main
def main():
    aplicar_estilo()
    TABELAS.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    print("\n[1] Leitura do treino e do teste já codificados")
    treino = carregar_parte("treino")
    teste = carregar_parte("teste")
    atributos = [c for c in treino.columns if c not in ("ALVO", "ANO")]
    X_tr, y_tr = treino[atributos], treino["ALVO"]
    X_te, y_te = teste[atributos], teste["ALVO"]
    print(f"  treino {X_tr.shape}  teste {X_te.shape}")

    pos_busca = indices_estratificados(y_tr, AMOSTRA_BUSCA, SEMENTE)
    X_busca, y_busca = X_tr.iloc[pos_busca], y_tr.iloc[pos_busca]

    linhas, por_classe, buscas, curvas = [], [], [], []
    ajustados, preds, scores, tempos = {}, {}, {}, {}

    def registrar(nome, familia, base, n_modelos, pred, extra):
        m = metricas_globais(y_te, pred)
        m.update({"modelo": nome, "familia": familia, "base": base,
                  "n_modelos": n_modelos, **extra})
        linhas.append(m)
        por_classe.append(metricas_por_classe(y_te, pred, nome))
        print(f"     teste f1_macro={m['f1_macro']:.3f}  acurácia={m['acuracia']:.3f}")

    print("\n[2] Algoritmos individuais (hiperparâmetros da Sprint 3)")
    cv_s3 = {}
    arq_s3 = TABELAS / "s3_metricas.csv"
    if arq_s3.exists():
        s3 = pd.read_csv(arq_s3).set_index("modelo")
        cv_s3 = s3[["f1_macro_cv", "f1_macro_cv_std"]].to_dict("index")
    for nome, spec in individuais().items():
        print(f"  -- {nome}")
        modelo, t_fit = treinar(spec["modelo"], X_tr, y_tr, spec["peso"])
        pred, t_pred = prever(modelo, X_te)
        ajustados[nome], preds[nome] = modelo, pred
        scores[nome] = pontuar(modelo, X_te)
        tempos[nome] = {"ajuste": t_fit, "previsao": t_pred}
        registrar(nome, "Individual", nome, 1, pred, {
            **cv_s3.get(nome, {}),
            "tempo_ajuste_s": t_fit, "tempo_total_s": t_fit,
            "tempo_previsao_s": t_pred,
            "hiperparametros": resumir_params(spec["modelo"].get_params(deep=False)
                                              if nome != "Rede neural" else
                                              {"hidden_layer_sizes": [128]}),
            "observacao": "Mesma configuração da Sprint 3, retreinada para medir tempo"})

    print("\n[3] Comitês homogêneos")
    passos = {1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400}
    for spec in homogeneos():
        print(f"  -- {spec['nome']}")
        melhores, linha_cv, t_busca, tab = buscar(spec, X_busca, y_busca)
        buscas.append(tab)
        print(f"     cv {linha_cv.f1_macro_cv:.3f}  params {melhores}  ({t_busca:.0f}s)")
        modelo = clone(spec["estimator"]).set_params(**melhores)
        modelo, t_fit = treinar(modelo, X_tr, y_tr, spec["peso"])
        pred, t_pred = prever(modelo, X_te)
        ajustados[spec["nome"]], preds[spec["nome"]] = modelo, pred
        tempos[spec["nome"]] = {"ajuste": t_fit, "previsao": t_pred}
        n = getattr(modelo, "n_iter_", None)
        n = int(n) if n is not None else len(getattr(modelo, "estimators_", []))
        registrar(spec["nome"], "Homogêneo", spec["base"], n, pred, {
            "f1_macro_cv": linha_cv.f1_macro_cv,
            "f1_macro_cv_std": linha_cv.f1_macro_cv_std,
            "tempo_busca_s": t_busca, "tempo_ajuste_s": t_fit,
            "tempo_total_s": t_fit, "tempo_previsao_s": t_pred,
            "hiperparametros": resumir_params({**spec["fixos"], **melhores}),
            "metodo": spec["metodo"],
            "observacao": "Número de iterações definido pela parada antecipada"
                          if spec["nome"] == "Gradient Boosting" else ""})
        curvas += curva_n_modelos(spec["nome"], modelo, X_te, y_te, passos)

    print("\n[4] Diversidade entre os individuais")
    disc, duplo = diversidade({n: preds[n] for n in CINCO}, y_te)
    disc.to_csv(TABELAS / "s4_discordancia.csv", encoding="utf-8-sig")
    duplo.to_csv(TABELAS / "s4_erro_duplo.csv", encoding="utf-8-sig")
    print((disc * 100).round(1))

    print(f"\n[5] Previsões fora do fold para o stacking ({FOLDS_EMPILHAMENTO} folds)")
    precisa_oof = sorted({b for h in heterogeneos() if h["tipo"] == "stacking"
                          for b in h["bases"]})
    specs_ind = individuais()
    specs_hom = {s["nome"]: s for s in homogeneos()}
    cv_oof = StratifiedKFold(n_splits=FOLDS_EMPILHAMENTO, shuffle=True,
                             random_state=SEMENTE)
    oof, t_oof = {}, {}
    for nome in precisa_oof:
        if nome in specs_ind:
            modelo, usa_peso = specs_ind[nome]["modelo"], specs_ind[nome]["peso"]
        else:
            modelo = clone(ajustados[nome])
            usa_peso = specs_hom[nome]["peso"]
        metodo = "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"
        params = {"sample_weight": pesos(y_tr)} if usa_peso else {}
        t0 = time.perf_counter()
        oof[nome] = cross_val_predict(clone(modelo), X_tr, y_tr, cv=cv_oof,
                                      method=metodo, params=params, n_jobs=1)
        t_oof[nome] = time.perf_counter() - t0
        if nome not in scores:
            scores[nome] = pontuar(ajustados[nome], X_te)
        print(f"  {nome:<28} {t_oof[nome]:.0f}s")

    print("\n[6] Comitês heterogêneos")
    classes = np.asarray(ajustados["Rede neural"].classes_, dtype=object)
    for spec in heterogeneos():
        print(f"  -- {spec['nome']}")
        bases = spec["bases"]
        t_bases = sum(tempos[b]["ajuste"] for b in bases)
        t_prev_bases = sum(tempos[b]["previsao"] for b in bases)
        extra = {"oraculo_teste": oraculo(preds, bases, y_te),
                 "discordancia_media": discordancia_media(preds, bases)}
        if spec["tipo"] == "voto":
            pred = votar([preds[b] for b in bases], bases)
            registrar(spec["nome"], "Heterogêneo", " + ".join(bases), len(bases), pred, {
                **extra, "tempo_ajuste_s": 0.0, "tempo_total_s": t_bases,
                "tempo_previsao_s": t_prev_bases,
                "hiperparametros": "voto majoritário; empate para o melhor na validação",
                "metodo": "Votação majoritária (hard voting)"})
        elif spec["tipo"] == "suave":
            pred = suave([scores[b] for b in bases], classes)
            registrar(spec["nome"], "Heterogêneo", " + ".join(bases), len(bases), pred, {
                **extra, "tempo_ajuste_s": 0.0, "tempo_total_s": t_bases,
                "tempo_previsao_s": t_prev_bases,
                "hiperparametros": "média simples das probabilidades",
                "metodo": "Votação suave (soft voting)"})
        else:
            Z_tr = np.hstack([oof[b] for b in bases])
            Z_te = np.hstack([scores[b] for b in bases])
            meta = meta_modelo()
            cv_meta = cross_val_score(meta, Z_tr, y_tr, scoring="f1_macro",
                                      cv=StratifiedKFold(N_FOLDS, shuffle=True,
                                                         random_state=SEMENTE))
            meta, t_meta = treinar(meta, Z_tr, y_tr, False)
            t0 = time.perf_counter()
            pred = meta.predict(Z_te)
            t_pred = time.perf_counter() - t0 + t_prev_bases
            t_oof_total = sum(t_oof[b] for b in bases)
            registrar(spec["nome"], "Heterogêneo", " + ".join(bases), len(bases), pred, {
                **extra, "f1_macro_cv": cv_meta.mean(), "f1_macro_cv_std": cv_meta.std(),
                "tempo_ajuste_s": t_meta + t_oof_total,
                "tempo_total_s": t_bases + t_oof_total + t_meta,
                "tempo_previsao_s": t_pred,
                "hiperparametros": json.dumps({
                    "meta_modelo": "regressão logística", "C": 1.0,
                    "class_weight": "balanced", "padronização": "z-score",
                    "folds_fora_do_fold": FOLDS_EMPILHAMENTO}, ensure_ascii=False),
                "metodo": "Stacking: meta-modelo treinado nas previsões fora do fold"})
            preds[spec["nome"]] = pred

    print("\n[7] Tabelas")
    tabela = pd.DataFrame(linhas)
    cols = ["modelo", "familia", "base", "n_modelos", "f1_macro_cv", "f1_macro_cv_std",
            "acuracia", "precisao_macro", "revocacao_macro", "f1_macro",
            "precisao_ponderada", "revocacao_ponderada", "f1_ponderado",
            "tempo_busca_s", "tempo_ajuste_s", "tempo_total_s", "tempo_previsao_s",
            "oraculo_teste", "discordancia_media", "hiperparametros", "metodo",
            "observacao"]
    tabela = tabela.reindex(columns=cols)
    tabela.to_csv(TABELAS / "s4_metricas.csv", index=False, encoding="utf-8-sig")
    classes_df = pd.concat(por_classe, ignore_index=True)
    classes_df.to_csv(TABELAS / "s4_por_classe.csv", index=False, encoding="utf-8-sig")
    pd.concat(buscas, ignore_index=True).to_csv(
        TABELAS / "s4_busca.csv", index=False, encoding="utf-8-sig")
    curva = pd.DataFrame(curvas)
    curva.to_csv(TABELAS / "s4_curva_n_modelos.csv", index=False, encoding="utf-8-sig")

    melhor = {fam: tabela[tabela.familia == fam].sort_values("f1_macro").iloc[-1]["modelo"]
              for fam in COR_FAMILIA}
    resumo_fam = tabela.groupby("familia").agg(
        melhor_f1_macro=("f1_macro", "max"), media_f1_macro=("f1_macro", "mean"),
        modelos=("modelo", "count")).reindex(list(COR_FAMILIA))
    resumo_fam["melhor_modelo"] = [melhor[f] for f in resumo_fam.index]
    resumo_fam.to_csv(TABELAS / "s4_resumo_familias.csv", encoding="utf-8-sig")
    print(tabela[["modelo", "familia", "f1_macro_cv", "f1_macro", "acuracia",
                  "tempo_total_s"]].round(3).to_string(index=False))

    print("\n[8] Gráficos")
    f1_ind = tabela[tabela.familia == "Individual"].set_index("modelo")["f1_macro"]
    grafico_geral(tabela)
    grafico_homogeneo_vs_base(tabela)
    if not curva.empty:
        grafico_curva(curva, f1_ind)
    grafico_diversidade(disc, duplo)
    grafico_custo(tabela)
    grafico_por_classe(classes_df, [melhor[f] for f in COR_FAMILIA])
    campeao = tabela.sort_values("f1_macro").iloc[-1]["modelo"]
    cm = grafico_confusao(y_te, preds[campeao], f"Matriz de confusão — {campeao}")
    pd.DataFrame(cm, index=CLASSES_ALVO, columns=CLASSES_ALVO).to_csv(
        TABELAS / "s4_confusao.csv", encoding="utf-8-sig")

    resumo = {
        "melhor_geral_no_teste": campeao,
        "melhor_por_familia_no_teste": melhor,
        "folds_busca": N_FOLDS,
        "amostra_busca": int(len(y_busca)),
        "folds_stacking": FOLDS_EMPILHAMENTO,
        "empilhou_comites": EMPILHAR_COMITES,
        "tempo_total_s": round(time.perf_counter() - t_total, 1),
    }
    (TABELAS / "s4_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPronto.", resumo)


if __name__ == "__main__":
    main()
