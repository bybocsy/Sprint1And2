"""Sprint 3: algoritmos individuais para classificar o tipo de resíduo.

A divisão 70/30 estratificada da Sprint 2 fica intocada. A validação é
cruzada (5 folds) dentro do treino. O teste só entra na avaliação final.

Rode a partir de src/, depois do pré-processamento:

    python sprint2_preprocessamento.py
    python sprint3_modelos.py
"""
import json
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import preprocessamento as pp
from config import (
    CINZA, CLASSES_ALVO, CORES, DADOS_PROCESSADOS, RAIZ, ROTULO_CURTO,
    SEMENTE, TABELAS, TEXTO_2,
)
from graficos import aplicar_estilo, fonte, num_br, salvar, subtitulo

MODELOS_DIR = RAIZ / "models" / "sprint3"
PASTA_FIG = "sprint3"
N_FOLDS = 5
AMOSTRA_BUSCA = 80_000
AMOSTRA_KNN_BUSCA = 12_000
AMOSTRA_KNN_TREINO = 25_000
AMOSTRA_KNN_TESTE = 20_000
FOLDS_KNN = 3

PREFIXOS_LUGAR = ("ZGL_", "REGIONAL_", "ORIGEM_")
PREFIXO_EMPRESA = ("TRANSPORTADORA_",)

GRUPOS_ATRIBUTO = [
    "CARGA_POR_TARA", "PESO_LIQUIDO", "DIA_MES", "DIA_SEMANA", "FIM_DE_SEMANA",
    "TRANSPORTADORA", "REGIONAL", "ORIGEM", "CHASSI", "TIPOCAD", "MES", "TARA", "ZGL",
]


def carregar_parte(nome):
    df = pd.read_csv(DADOS_PROCESSADOS / f"{nome}.csv.gz")
    df["ANO"] = df["ANO"].astype(int)
    atributos = [c for c in df.columns if c not in ("ALVO", "ANO")]
    df[atributos] = df[atributos].astype(np.float32)
    return df


def indices_estratificados(y, n, semente):
    n = min(int(n), len(y))
    pos, _ = train_test_split(
        np.arange(len(y)), train_size=n, stratify=y, random_state=semente)
    return pos


def grade():
    """Hiperparâmetros buscados. O restante fica fixo e entra no relatório."""
    return [
        {
            "nome": "Regressão logística",
            "amostra": "busca",
            "estimator": LogisticRegression(
                solver="lbfgs", max_iter=250, tol=1e-3,
                class_weight="balanced", random_state=SEMENTE),
            "param_grid": {"C": [0.1, 1.0, 10.0]},
            "fixos": {"solver": "lbfgs", "class_weight": "balanced", "max_iter": 250},
        },
        {
            "nome": "Árvore de decisão",
            "amostra": "busca",
            "estimator": DecisionTreeClassifier(
                class_weight="balanced", random_state=SEMENTE),
            "param_grid": {
                "max_depth": [12, 24, None],
                "min_samples_leaf": [50],
            },
            "fixos": {"criterion": "gini", "class_weight": "balanced",
                      "min_samples_leaf": 50},
        },
        {
            "nome": "Naive Bayes",
            "amostra": "busca",
            "estimator": GaussianNB(),
            "param_grid": {"var_smoothing": [1e-9, 1e-5]},
            "fixos": {},
        },
        {
            "nome": "SVM linear",
            "amostra": "busca",
            "estimator": LinearSVC(
                class_weight="balanced", dual="auto", max_iter=2000,
                random_state=SEMENTE),
            "param_grid": {"C": [0.1, 1.0, 10.0]},
            "fixos": {"class_weight": "balanced", "loss": "squared_hinge",
                      "multi_class": "ovr"},
        },
        {
            "nome": "Rede neural",
            "amostra": "busca",
            "estimator": MLPClassifier(
                activation="relu", solver="adam", learning_rate_init=1e-3,
                batch_size=512, max_iter=25, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=4,
                random_state=SEMENTE),
            "param_grid": {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-4]},
            "fixos": {
                "activation": "relu", "solver": "adam",
                "learning_rate_init": 0.001, "batch_size": 512,
                "early_stopping": True, "validation_fraction": 0.1, "alpha": 1e-4,
            },
            "sample_weight": True,
        },
        {
            "nome": "KNN",
            "amostra": "knn",
            "estimator": KNeighborsClassifier(algorithm="brute", n_jobs=1),
            "param_grid": {"n_neighbors": [7, 21], "weights": ["distance"]},
            "fixos": {"metric": "minkowski", "p": 2, "algorithm": "brute"},
        },
    ]


def resumir_params(params):
    texto = {}
    for k, v in params.items():
        texto[k] = list(v) if isinstance(v, tuple) else v
    return json.dumps(texto, ensure_ascii=False)


def metricas_globais(y_true, y_pred):
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao_macro": precision_score(y_true, y_pred, average="macro",
                                          zero_division=0),
        "revocacao_macro": recall_score(y_true, y_pred, average="macro",
                                         zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precisao_ponderada": precision_score(y_true, y_pred, average="weighted",
                                               zero_division=0),
        "revocacao_ponderada": recall_score(y_true, y_pred, average="weighted",
                                             zero_division=0),
        "f1_ponderado": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def metricas_por_classe(y_true, y_pred, modelo):
    p = precision_score(y_true, y_pred, labels=CLASSES_ALVO, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=CLASSES_ALVO, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=CLASSES_ALVO, average=None, zero_division=0)
    suporte = pd.Series(y_true).value_counts()
    return pd.DataFrame({
        "modelo": modelo,
        "classe": CLASSES_ALVO,
        "precisao": p,
        "revocacao": r,
        "f1": f,
        "suporte": [int(suporte.get(c, 0)) for c in CLASSES_ALVO],
    })


def buscar(spec, X, y):
    folds = N_FOLDS if spec["amostra"] == "busca" else FOLDS_KNN
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
    busca = GridSearchCV(
        spec["estimator"], spec["param_grid"], scoring="f1_macro", cv=cv,
        n_jobs=-1, refit=True, error_score="raise",
    )
    fit_params = {}
    if spec.get("sample_weight"):
        fit_params["sample_weight"] = compute_sample_weight("balanced", y)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The line search", category=UserWarning)
        busca.fit(X, y, **fit_params)
    segundos = time.perf_counter() - t0
    linhas = []
    for params, media, desvio, rank in zip(
            busca.cv_results_["params"],
            busca.cv_results_["mean_test_score"],
            busca.cv_results_["std_test_score"],
            busca.cv_results_["rank_test_score"]):
        linhas.append({
            "modelo": spec["nome"],
            "params": resumir_params(params),
            "f1_macro_cv": media,
            "f1_macro_cv_std": desvio,
            "rank": int(rank),
            "folds": folds,
            "linhas_busca": len(y),
        })
    return busca, segundos, pd.DataFrame(linhas)


def ajustar(spec, busca, X, y):
    modelo = clone(busca.best_estimator_)
    fit_params = {}
    if spec.get("sample_weight"):
        fit_params["sample_weight"] = compute_sample_weight("balanced", y)
    t0 = time.perf_counter()
    modelo.fit(X, y, **fit_params)
    return modelo, time.perf_counter() - t0


def colunas_sem(X, prefixos):
    manter = [c for c in X.columns if not c.startswith(prefixos)]
    return X[manter]


def grupo_atributo(coluna):
    for grupo in GRUPOS_ATRIBUTO:
        if coluna == grupo or coluna.startswith(grupo + "_"):
            return grupo
    return coluna


def importancia_agrupada(modelo, colunas):
    if hasattr(modelo, "feature_importances_"):
        valores = np.asarray(modelo.feature_importances_, dtype=float)
    elif hasattr(modelo, "coef_"):
        valores = np.abs(np.asarray(modelo.coef_, dtype=float)).mean(axis=0)
    else:
        return None
    serie = pd.Series(valores, index=colunas).groupby(grupo_atributo).sum()
    serie = serie / serie.sum()
    return serie.sort_values(ascending=False)


def baseline_majoritaria(y_tr, y_te):
    classe = y_tr.value_counts().idxmax()
    pred = np.full(len(y_te), classe)
    return pred, classe


def baseline_zona(X_tr, y_tr, X_te):
    def zona(X):
        cols = [c for c in X.columns if c.startswith("ZGL_")]
        cod = X[cols].to_numpy()
        return np.asarray(cols)[cod.argmax(axis=1)]

    tab = pd.DataFrame({"zona": zona(X_tr), "alvo": y_tr.to_numpy()})
    mapa = tab.groupby("zona")["alvo"].agg(lambda s: s.value_counts().idxmax())
    pred = pd.Series(zona(X_te)).map(mapa)
    falta = pred.isna()
    if falta.any():
        pred[falta] = y_tr.value_counts().idxmax()
    return pred.to_numpy()


def grafico_f1(tabela):
    d = tabela.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cores = [CORES[0] if n != "Classe majoritária" else CINZA for n in d.modelo]
    y = np.arange(len(d))
    ax.barh(y, d.f1_macro, color=cores, height=0.62)
    for yi, v in zip(y, d.f1_macro):
        ax.text(v + 0.01, yi, num_br(v * 100) + "%", va="center", fontsize=8.5, color=TEXTO_2)
    ax.set_yticks(y, d.modelo)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 macro no teste")
    ax.grid(axis="y", visible=False)
    ax.set_title("F1 macro no teste, com as classes no mesmo peso", pad=22)
    subtitulo(ax, "A classe majoritária é a linha de base: chutar sempre a classe mais comum.")
    fonte(fig, "Fonte: experimentos da Sprint 3. Métrica calculada no teste de 30%.")
    salvar(fig, "01_f1_macro", PASTA_FIG)


def grafico_f1_classe(por_classe, modelos):
    d = por_classe[por_classe.modelo.isin(modelos)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(CLASSES_ALVO))
    largura = 0.8 / len(modelos)
    for i, nome in enumerate(modelos):
        vals = d[d.modelo == nome].set_index("classe").reindex(CLASSES_ALVO)["f1"]
        ax.bar(x + (i - (len(modelos) - 1) / 2) * largura, vals, width=largura * 0.92,
               color=CORES[i % len(CORES)], label=nome)
    ax.set_xticks(x, [ROTULO_CURTO[c] for c in CLASSES_ALVO], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("F1 de cada classe no teste", pad=22)
    subtitulo(ax, "Varrição é a classe rara. Macro dá o mesmo peso a ela e a Domiciliar.")
    fonte(fig, "Fonte: experimentos da Sprint 3, conjunto de teste.")
    salvar(fig, "02_f1_por_classe", PASTA_FIG)


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
    fonte(fig, "Fonte: Sprint 3. Cada linha soma 100% da classe verdadeira.", y=-0.08)
    salvar(fig, "03_matriz_confusao", PASTA_FIG)
    return cm


def grafico_importancia(serie):
    d = serie.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(np.arange(len(d)), d.values, color=CORES[0], height=0.62)
    ax.set_yticks(np.arange(len(d)), d.index)
    ax.set_xlabel("Participação na importância (soma = 1)")
    ax.grid(axis="y", visible=False)
    ax.set_title("De onde a árvore tira a decisão", pad=22)
    subtitulo(ax, "Importâncias das colunas one-hot somadas de volta no atributo original.")
    fonte(fig, "Fonte: árvore de decisão treinada no conjunto de treino.")
    salvar(fig, "04_importancia", PASTA_FIG)


def grafico_ablacao(ablacao):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(ablacao))[::-1]
    ax.barh(y, ablacao.f1_macro, color=CORES[1], height=0.6)
    for yi, v in zip(y, ablacao.f1_macro):
        ax.text(v + 0.01, yi, num_br(v * 100) + "%", va="center", fontsize=8.5, color=TEXTO_2)
    ax.set_yticks(y, ablacao.variante)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 macro no teste")
    ax.grid(axis="y", visible=False)
    ax.set_title("O modelo depende do lugar e da empresa?", pad=22)
    subtitulo(ax, "Mesmo algoritmo e mesmos hiperparâmetros. Só muda o conjunto de atributos.")
    fonte(fig, "Fonte: Sprint 3, conjunto de teste.")
    salvar(fig, "05_ablacao", PASTA_FIG)


def grafico_por_ano(por_ano):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(por_ano))
    ax.bar(x, por_ano.f1_macro, color=CORES[0], width=0.55)
    for xi, v, n in zip(x, por_ano.f1_macro, por_ano.linhas):
        ax.text(xi, v + 0.02, f"{num_br(v * 100)}%\n{num_br(n, 0)} linhas",
                ha="center", fontsize=8, color=TEXTO_2)
    ax.set_xticks(x, por_ano.ano.astype(str))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 macro")
    ax.grid(axis="x", visible=False)
    ax.set_title("O melhor modelo, separado por ano da pesagem", pad=22)
    subtitulo(ax, "O ano não entra como atributo. 2024 veio só com cadastros ativos.")
    fonte(fig, "Fonte: Sprint 3, recorte do conjunto de teste.")
    salvar(fig, "06_f1_por_ano", PASTA_FIG)


def main():
    aplicar_estilo()
    TABELAS.mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    print("\n[1] Leitura do treino e do teste já codificados")
    treino = carregar_parte("treino")
    teste = carregar_parte("teste")
    atributos = [c for c in treino.columns if c not in ("ALVO", "ANO")]
    X_tr, y_tr = treino[atributos], treino["ALVO"]
    X_te, y_te, ano_te = teste[atributos], teste["ALVO"], teste["ANO"]
    print(f"  treino {X_tr.shape}  teste {X_te.shape}  atributos {len(atributos)}")

    print("\n[2] Proporções da divisão")
    n_tr, n_te = len(treino), len(teste)
    n_busca = min(AMOSTRA_BUSCA, n_tr)
    n_knn_busca = min(AMOSTRA_KNN_BUSCA, n_tr)
    n_knn_treino = min(AMOSTRA_KNN_TREINO, n_tr)
    n_knn_teste = min(AMOSTRA_KNN_TESTE, n_te)
    divisao = pd.DataFrame([
        {"parte": "Treino (ajuste final)", "linhas": n_tr,
         "percentual": n_tr / (n_tr + n_te)},
        {"parte": "Teste (avaliação final)", "linhas": n_te,
         "percentual": n_te / (n_tr + n_te)},
        {"parte": "Amostra de busca (5 modelos)", "linhas": n_busca,
         "percentual": n_busca / (n_tr + n_te)},
        {"parte": "Treino por fold da busca (aprox. 4/5)", "linhas": n_busca - n_busca // N_FOLDS,
         "percentual": (n_busca - n_busca // N_FOLDS) / (n_tr + n_te)},
        {"parte": "Validação por fold da busca (aprox. 1/5)", "linhas": n_busca // N_FOLDS,
         "percentual": (n_busca // N_FOLDS) / (n_tr + n_te)},
        {"parte": "Amostra de busca do KNN (3 folds)", "linhas": n_knn_busca,
         "percentual": n_knn_busca / (n_tr + n_te)},
        {"parte": "Treino por fold da busca do KNN (aprox. 2/3)",
         "linhas": n_knn_busca - n_knn_busca // FOLDS_KNN,
         "percentual": (n_knn_busca - n_knn_busca // FOLDS_KNN) / (n_tr + n_te)},
        {"parte": "Validação por fold da busca do KNN (aprox. 1/3)",
         "linhas": n_knn_busca // FOLDS_KNN,
         "percentual": (n_knn_busca // FOLDS_KNN) / (n_tr + n_te)},
        {"parte": "Treino final do KNN", "linhas": n_knn_treino,
         "percentual": n_knn_treino / (n_tr + n_te)},
        {"parte": "Amostra de teste do KNN", "linhas": n_knn_teste,
         "percentual": n_knn_teste / (n_tr + n_te)},
    ])
    divisao.to_csv(TABELAS / "s3_divisao.csv", index=False, encoding="utf-8-sig")

    prop = pd.DataFrame({
        "treino": y_tr.value_counts(normalize=True),
        "teste": y_te.value_counts(normalize=True),
    }).reindex(CLASSES_ALVO)
    prop.to_csv(TABELAS / "s3_proporcao_classes.csv", encoding="utf-8-sig")
    print(prop.round(4))

    print("\n[3] Linhas de base")
    pred_maj, classe_maj = baseline_majoritaria(y_tr, y_te)
    pred_zona = baseline_zona(X_tr, y_tr, X_te)
    resultados = []
    por_classe = []
    for nome, pred, nota in [
        ("Classe majoritária", pred_maj, f"Sempre prevê {classe_maj}"),
        ("Zona (maioria da ZGL no treino)", pred_zona, "Não é um algoritmo treinado; mostra o quanto a zona sozinha entrega"),
    ]:
        m = metricas_globais(y_te, pred)
        m.update({"modelo": nome, "f1_macro_cv": np.nan, "f1_macro_cv_std": np.nan,
                  "tempo_busca_s": 0, "tempo_ajuste_s": 0, "hiperparametros": "",
                  "observacao": nota})
        resultados.append(m)
        por_classe.append(metricas_por_classe(y_te, pred, nome))
        print(f"  {nome:<40} f1_macro={m['f1_macro']:.3f}  acurácia={m['acuracia']:.3f}")

    print("\n[4] Busca de hiperparâmetros e ajuste final")
    pos_busca = indices_estratificados(y_tr, AMOSTRA_BUSCA, SEMENTE)
    X_busca, y_busca = X_tr.iloc[pos_busca], y_tr.iloc[pos_busca]
    pos_knn_te = indices_estratificados(y_te, AMOSTRA_KNN_TESTE, SEMENTE)
    buscas = []
    previsoes = {}
    modelos = {}

    for spec in grade():
        print(f"\n  -- {spec['nome']}")
        if spec["amostra"] == "busca":
            Xb, yb = X_busca, y_busca
        else:
            pos = indices_estratificados(y_tr, AMOSTRA_KNN_BUSCA, SEMENTE + 1)
            Xb, yb = X_tr.iloc[pos], y_tr.iloc[pos]
        busca, tempo_busca, tabela_busca = buscar(spec, Xb, yb)
        buscas.append(tabela_busca)
        print(f"     cv {busca.best_score_:.3f}  params {busca.best_params_}  "
              f"({tempo_busca:.0f}s)")

        if spec["amostra"] == "busca":
            modelo, tempo_ajuste = ajustar(spec, busca, X_tr, y_tr)
            pred = modelo.predict(X_te)
            modelos[spec["nome"]] = modelo
        else:
            pos_fit = indices_estratificados(y_tr, AMOSTRA_KNN_TREINO, SEMENTE + 2)
            modelo, tempo_ajuste = ajustar(spec, busca, X_tr.iloc[pos_fit], y_tr.iloc[pos_fit])
            pred_sub = modelo.predict(X_te.iloc[pos_knn_te])
            pred = np.array([None] * len(y_te), dtype=object)
            pred[pos_knn_te] = pred_sub
            modelos[spec["nome"]] = modelo

        previsoes[spec["nome"]] = pred
        if spec["amostra"] == "busca":
            y_eval, p_eval = y_te, pred
            nota = "Hiperparâmetros pela CV na amostra; ajuste final no treino inteiro"
        else:
            y_eval = y_te.iloc[pos_knn_te]
            p_eval = pred[pos_knn_te]
            nota = (f"Treino estratificado de {AMOSTRA_KNN_TREINO} linhas; "
                    f"métricas numa amostra estratificada de {AMOSTRA_KNN_TESTE} do teste")
        m = metricas_globais(y_eval, p_eval)
        escolhidos = {**spec["fixos"], **busca.best_params_}
        m.update({
            "modelo": spec["nome"],
            "f1_macro_cv": busca.best_score_,
            "f1_macro_cv_std": float(tabela_busca.loc[tabela_busca["rank"] == 1, "f1_macro_cv_std"].iloc[0]),
            "tempo_busca_s": round(tempo_busca, 1),
            "tempo_ajuste_s": round(tempo_ajuste, 1),
            "hiperparametros": resumir_params(escolhidos),
            "observacao": nota,
        })
        resultados.append(m)
        por_classe.append(metricas_por_classe(y_eval, p_eval, spec["nome"]))
        print(f"     teste f1_macro={m['f1_macro']:.3f}  acurácia={m['acuracia']:.3f}  "
              f"ajuste {tempo_ajuste:.0f}s")

    tabela = pd.DataFrame(resultados)
    cols_frente = ["modelo", "f1_macro_cv", "f1_macro_cv_std", "acuracia",
                   "precisao_macro", "revocacao_macro", "f1_macro",
                   "precisao_ponderada", "revocacao_ponderada", "f1_ponderado",
                   "tempo_busca_s", "tempo_ajuste_s", "hiperparametros", "observacao"]
    tabela[cols_frente].to_csv(TABELAS / "s3_metricas.csv", index=False, encoding="utf-8-sig")
    classes = pd.concat(por_classe, ignore_index=True)
    classes.to_csv(TABELAS / "s3_por_classe.csv", index=False, encoding="utf-8-sig")
    pd.concat(buscas, ignore_index=True).to_csv(
        TABELAS / "s3_busca.csv", index=False, encoding="utf-8-sig")

    comparaveis = tabela[~tabela.modelo.isin(
        ["Classe majoritária", "Zona (maioria da ZGL no treino)", "KNN"])]
    melhor_nome = comparaveis.sort_values("f1_macro_cv", ascending=False).iloc[0]["modelo"]
    print(f"\n  melhor pela validação cruzada: {melhor_nome}")

    print("\n[5] Mesma amostra do teste, para comparar o KNN")
    linhas_par = []
    y_par = y_te.iloc[pos_knn_te]
    for nome, pred in previsoes.items():
        if nome == "KNN":
            p = pred[pos_knn_te]
        else:
            p = np.asarray(pred)[pos_knn_te]
        m = metricas_globais(y_par, p)
        m["modelo"] = nome
        linhas_par.append(m)
    pd.DataFrame(linhas_par).to_csv(TABELAS / "s3_amostra_pareada.csv", index=False, encoding="utf-8-sig")

    print("\n[6] Ablação de zona e transportadora")
    spec_melhor = next(s for s in grade() if s["nome"] == melhor_nome)
    variantes = [
        ("Todos os atributos", X_tr, X_te),
        ("Sem transportadora", colunas_sem(X_tr, PREFIXO_EMPRESA), colunas_sem(X_te, PREFIXO_EMPRESA)),
        ("Sem ZGL, regional e origem", colunas_sem(X_tr, PREFIXOS_LUGAR), colunas_sem(X_te, PREFIXOS_LUGAR)),
        ("Sem lugar e sem transportadora",
         colunas_sem(X_tr, PREFIXOS_LUGAR + PREFIXO_EMPRESA),
         colunas_sem(X_te, PREFIXOS_LUGAR + PREFIXO_EMPRESA)),
    ]
    # O modelo completo já foi treinado. Reaproveita a previsão.
    ab_linhas = []
    pred_completa = previsoes[melhor_nome]
    ab_linhas.append({"variante": "Todos os atributos", "modelo": melhor_nome,
                      **metricas_globais(y_te, pred_completa)})
    ja_ajustado = type("Busca", (), {"best_estimator_": modelos[melhor_nome]})()
    for nome_var, Xa, Xe in variantes[1:]:
        print(f"  {nome_var}")
        modelo, _ = ajustar(spec_melhor, ja_ajustado, Xa, y_tr)
        pred = modelo.predict(Xe)
        ab_linhas.append({"variante": nome_var, "modelo": melhor_nome,
                          **metricas_globais(y_te, pred)})
    ablacao = pd.DataFrame(ab_linhas)
    ablacao.to_csv(TABELAS / "s3_ablacao.csv", index=False, encoding="utf-8-sig")
    print(ablacao[["variante", "f1_macro", "acuracia"]].round(3))

    print("\n[7] Resultado por ano e importância")
    pred_melhor = np.asarray(previsoes[melhor_nome])
    por_ano = []
    y_np = y_te.to_numpy()
    ano_np = ano_te.to_numpy()
    for ano in sorted(ano_te.unique()):
        mask = ano_np == ano
        m = metricas_globais(y_np[mask], pred_melhor[mask])
        por_ano.append({"ano": int(ano), "linhas": int(mask.sum()), **m})
    por_ano = pd.DataFrame(por_ano)
    por_ano.to_csv(TABELAS / "s3_por_ano.csv", index=False, encoding="utf-8-sig")

    arvore = modelos.get("Árvore de decisão")
    imp = importancia_agrupada(arvore, atributos) if arvore is not None else None
    if imp is not None:
        imp.rename("importancia").to_csv(TABELAS / "s3_importancia.csv", encoding="utf-8-sig")

    cm = confusion_matrix(y_te, pred_melhor, labels=CLASSES_ALVO)
    pd.DataFrame(cm, index=CLASSES_ALVO, columns=CLASSES_ALVO).to_csv(
        TABELAS / "s3_confusao.csv", encoding="utf-8-sig")

    print("\n[8] Gráficos")
    ordem_grafico = [n for n in tabela.modelo if n != "KNN"]
    grafico_f1(tabela[tabela.modelo.isin(ordem_grafico)])
    destaque = [melhor_nome, "Árvore de decisão", "Regressão logística"]
    destaque = list(dict.fromkeys(destaque))
    grafico_f1_classe(classes, destaque)
    grafico_confusao(y_te, pred_melhor, f"Matriz de confusão — {melhor_nome}")
    if imp is not None:
        grafico_importancia(imp)
    grafico_ablacao(ablacao)
    grafico_por_ano(por_ano)

    resumo = {
        "melhor_pela_cv": melhor_nome,
        "n_treino": int(n_tr),
        "n_teste": int(n_te),
        "n_atributos": len(atributos),
        "n_folds": N_FOLDS,
        "amostra_busca": int(len(y_busca)),
        "classe_majoritaria": classe_maj,
        "tempo_total_s": round(time.perf_counter() - t_total, 1),
    }
    (TABELAS / "s3_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPronto.", resumo)


if __name__ == "__main__":
    main()
