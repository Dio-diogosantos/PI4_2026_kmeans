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
