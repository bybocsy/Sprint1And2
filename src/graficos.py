"""Estilo único para todos os gráficos do projeto (matplotlib + seaborn)."""
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from config import AZUL_SEQ, CINZA, FIGURAS, TEXTO, TEXTO_2

FUNDO = "#fcfcfb"
GRADE = "#e4e3df"

MAPA_SEQUENCIAL = LinearSegmentedColormap.from_list("azul", AZUL_SEQ)
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_vermelho", ["#b8302f", "#ef8a89", "#f0efec", "#6da7ec", "#184f95"])


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": FUNDO,
        "axes.facecolor": FUNDO,
        "savefig.facecolor": FUNDO,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": TEXTO_2,
        "axes.edgecolor": GRADE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2,
        "text.color": TEXTO,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def milhar(x, _pos=None):
    return f"{x:,.0f}".replace(",", ".")


def num_br(x, casas=1):
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def eixo_milhar(ax, eixo="y"):
    alvo = ax.yaxis if eixo == "y" else ax.xaxis
    alvo.set_major_formatter(mticker.FuncFormatter(milhar))


def subtitulo(ax, texto):
    ax.text(0, 1.01, texto, transform=ax.transAxes, fontsize=9,
            color=TEXTO_2, va="bottom")


def fonte(fig, texto="Fonte: Prefeitura de Fortaleza - pesagens no ASMOC, 2022-2024.", y=-0.02):
    fig.text(0.01, y, texto, fontsize=7.5, color=CINZA, ha="left", va="top")


MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun",
               "jul", "ago", "set", "out", "nov", "dez"]


def eixo_datas(ax, intervalo=4):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, intervalo)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _p: (lambda d: f"{MESES_ABREV[d.month - 1]}/{d:%y}")(mdates.num2date(x))))


def salvar(fig, nome, pasta="sprint1"):
    destino = FIGURAS / pasta
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.png"
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print("  figura salva:", caminho.relative_to(FIGURAS.parents[1]))
    return caminho
