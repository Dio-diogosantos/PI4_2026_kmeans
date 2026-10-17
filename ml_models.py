import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def calcular_metricas_k(X_scaled, max_k=5) -> pd.DataFrame:
    """Calcula as métricas de validação estatística para valores de k de 2 até max_k."""
    metricas = []
    for k_val in range(2, max_k + 1):
        km = KMeans(
            n_clusters=k_val,
            init="k-means++",
            n_init=25,
            max_iter=500,
            tol=0.00005,
            random_state=42,
        )
        labels = km.fit_predict(X_scaled)
        metricas.append(
            {
                "k": k_val,
                "Silhouette (↑)": round(
                    silhouette_score(X_scaled, labels), 3
                ),
                "Davies-Bouldin (↓)": round(
                    davies_bouldin_score(X_scaled, labels), 3
                ),
                "Calinski-Harabasz (↑)": round(
                    calinski_harabasz_score(X_scaled, labels), 1
                ),
            }
        )
    return pd.DataFrame(metricas)


def sugerir_k_recomendado(tabela_metricas: pd.DataFrame) -> int:
    """Calcula o k ideal somando os ranks das três métricas de validação."""
    tabela_ranqueada = tabela_metricas.copy()

    tabela_ranqueada["rank_silhouette"] = tabela_ranqueada[
        "Silhouette (↑)"
    ].rank(ascending=False, method="min")
    tabela_ranqueada["rank_davies"] = tabela_ranqueada[
        "Davies-Bouldin (↓)"
    ].rank(ascending=True, method="min")
    tabela_ranqueada["rank_calinski"] = tabela_ranqueada[
        "Calinski-Harabasz (↑)"
    ].rank(ascending=False, method="min")

    tabela_ranqueada["score_recomendacao"] = (
        tabela_ranqueada["rank_silhouette"]
        + tabela_ranqueada["rank_davies"]
        + tabela_ranqueada["rank_calinski"]
    )

    melhor_k = tabela_ranqueada.sort_values(
        by=[
            "score_recomendacao",
            "rank_silhouette",
            "rank_davies",
            "rank_calinski",
        ],
        ascending=[True, True, True, True],
    )["k"].iloc[0]

    return int(melhor_k)



import numpy as np
import pandas as pd


def rotular_perfis(
    df_clusterizado: pd.DataFrame,
    nome_col_grupo: str = "Grupo",
    col_media: str = "Média Global",
    col_faltas: str = "Faltas Médias",
) -> dict[int, str]:
    """Rotula pedagogicamente os clusters ('Avançado', 'Regular', 'Necessita de Reforço')

    baseado no desempenho das médias e faltas.
    """
    # Agrupa e calcula as médias por cluster
    resumo = (
        df_clusterizado.groupby(nome_col_grupo)
        .agg(
            media_global=(
                col_media
                if col_media in df_clusterizado.columns
                else df_clusterizado.select_dtypes(
                    include=["number"]
                ).columns[0],
                "mean",
            ),
            faltas_medias=(
                col_faltas
                if col_faltas in df_clusterizado.columns
                else df_clusterizado.select_dtypes(
                    include=["number"]
                ).columns[-1],
                "mean",
            ),
        )
        .reset_index()
    )

    desvio_media = resumo["media_global"].std(ddof=0)
    desvio_media = desvio_media if desvio_media != 0 else 1.0

    desvio_faltas = resumo["faltas_medias"].std(ddof=0)
    desvio_faltas = desvio_faltas if desvio_faltas != 0 else 1.0

    # Score: Maior Média e Menos Faltas geram maior Score
    resumo["score_perfil"] = (
        (resumo["media_global"] - resumo["media_global"].mean()) / desvio_media
    ) - (
        (resumo["faltas_medias"] - resumo["faltas_medias"].mean()) / desvio_faltas
    )

    resumo = resumo.sort_values(
        by=["score_perfil", "media_global", "faltas_medias"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    total_clusters = len(resumo)
    mapeamento = {}

    for posicao, grupo in enumerate(resumo[nome_col_grupo].tolist()):
        if posicao == 0:
            mapeamento[int(grupo)] = "Avançado"
        elif posicao == total_clusters - 1:
            mapeamento[int(grupo)] = "Necessita de Reforço"
        elif total_clusters == 3:
            mapeamento[int(grupo)] = "Regular"
        else:
            mapeamento[int(grupo)] = f"Regular {posicao}"

    return mapeamento



import plotly.graph_objects as go
from plotly.subplots import make_subplots


def criar_grafico_impacto_categoria(
    df: pd.DataFrame,
    coluna_categoria: str,
    col_media: str = "Média Global",
    col_faltas: str = "Faltas Médias",
):
    """Gera um gráfico comparativo de barras de Média Global e Faltas Médias

    agrupados por uma variável categórica/nominal.
    """
    resumo = (
        df.groupby(coluna_categoria)[[col_media, col_faltas]]
        .mean()
        .reset_index()
    )

    # Cria subplots (2 colunas: Média Global e Faltas)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{coluna_categoria} x {col_media}",
            f"{coluna_categoria} x {col_faltas}",
        ],
    )

    # Gráfico 1: Média Global
    fig.add_trace(
        go.Bar(
            x=resumo[coluna_categoria].astype(str),
            y=resumo[col_media],
            name=col_media,
            text=resumo[col_media].round(2),
            textposition="auto",
            marker_color="#4682B4",
        ),
        row=1,
        col=1,
    )

    # Gráfico 2: Faltas Médias
    fig.add_trace(
        go.Bar(
            x=resumo[coluna_categoria].astype(str),
            y=resumo[col_faltas],
            name=col_faltas,
            text=resumo[col_faltas].round(2),
            textposition="auto",
            marker_color="#5F9EA0",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text=f"Impacto de '{coluna_categoria}' no Desempenho e na Assiduidade",
        showlegend=False,
        height=400,
    )

    return fig, resumo
