import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="Painel Educacional - Fundamental 2", layout="wide")
st.title("Painel de Segmentação Escolar (K-Means)")

# 1. Controles na Barra Lateral: Ano, Etapa e K
st.sidebar.header("Seleção de Filtros")

# Seletor do Ano Letivo
anos_disponiveis = [2025, 2024, 2023]
ano_escolhido = st.sidebar.selectbox("Escolha o Ano Letivo:", anos_disponiveis)

# Seletor da Etapa
etapa_opcoes = {
    "6º ano": 6,
    "7º ano": 7,
    "8º ano": 8,
    "9º ano": 9
}
etapa_nome = st.sidebar.selectbox("Escolha a Etapa:", list(etapa_opcoes.keys()))
etapa_valor = etapa_opcoes[etapa_nome]

# Seletor de Clusters
k_escolhido = st.sidebar.slider("Quantidade de clusters (k):", min_value=2, max_value=5, value=3, step=1)
st.sidebar.info(f"Filtro ativo: **{ano_escolhido}** | **{etapa_nome}** | **k = {k_escolhido}**")

# 2. Carregamento e Processamento Parametrizado por (Ano, Etapa)
@st.cache_data
def carregar_e_processar(ano, etapa):
    caminho = 'Dados Fundamental 2 Limpos 2023, 2024 e 2025.xlsx'
    df_completo = pd.read_excel(caminho).dropna(how='all')

    # Filtra simultaneamente por Ano e por Etapa_1
    df = df_completo[(df_completo['Ano'] == ano) & (df_completo['Etapa_1'] == etapa)].copy()

    # As 22 variáveis padronizadas do treinamento
    features_treino = [
        'Transporte_1', 'BF_1', 'M_Leitura', 'M_Registro', 'M_Gênero', 'M_Coecoesão',
        'M_Português', 'F_Português', 'M_Matemática', 'F_Matemática', 'M_Ciências', 'F_Ciências',
        'M_História', 'F_História', 'M_Geografia', 'F_Geografia', 'M_Arte', 'F_Arte',
        'M_EdFísica', 'F_EdFísica', 'M_Inglês', 'F_Inglês'
    ]

    # Mantém apenas as features que estão presentes na base
    features_treino = [f for f in features_treino if f in df.columns]

    # Imputação KNN (k=5)
    imputer = KNNImputer(n_neighbors=5)
    df[features_treino] = imputer.fit_transform(df[features_treino])

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_treino])

    # Curva do Cotovelo para o subgrupo filtrado
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    # Tabela comparativa de validação (k=2 a 5)
    metricas = []
    for k_val in range(2, 6):
        km = KMeans(n_clusters=k_val, init='k-means++', n_init=25, max_iter=500, tol=0.00005, random_state=42)
        labels = km.fit_predict(X_scaled)
        metricas.append({
            'k': k_val,
            'Silhouette (↑)': round(silhouette_score(X_scaled, labels), 3),
            'Davies-Bouldin (↓)': round(davies_bouldin_score(X_scaled, labels), 3),
            'Calinski-Harabasz (↑)': round(calinski_harabasz_score(X_scaled, labels), 1)
        })
    tabela_validacao = pd.DataFrame(metricas)

    return df, X_scaled, wcss, tabela_validacao, features_treino

# Executa para o corte (Ano, Etapa) selecionado
dados_filtrados, X_scaled, wcss, tabela_validacao, features_treino = carregar_e_processar(ano_escolhido, etapa_valor)

# 3. Treina o K-Means com o k selecionado
kmeans = KMeans(n_clusters=k_escolhido, init='k-means++', n_init=25, max_iter=500, tol=0.00005, random_state=42)
dados_filtrados['Grupo'] = kmeans.fit_predict(X_scaled)

# 4. Tabela de Agregação por Perfil
dict_agg = {col: 'mean' for col in features_treino}
if 'Média Global' in dados_filtrados.columns:
    dict_agg['Média Global'] = 'mean'
if 'Faltas Médias' in dados_filtrados.columns:
    dict_agg['Faltas Médias'] = 'mean'
if 'Matrícula' in dados_filtrados.columns:
    dict_agg['Matrícula'] = 'count'

analise_perfis = dados_filtrados.groupby('Grupo').agg(dict_agg)
if 'Matrícula' in analise_perfis.columns:
    analise_perfis = analise_perfis.rename(columns={'Matrícula': 'Qtd_Alunos'})

# 5. Estrutura nas 5 Abas solicitadas
aba_decisao, aba_heatmap, aba_perfis, aba_dispersao, aba_estudantes = st.tabs([
    "1. Decisão do K (Métricas & Cotovelo)",
    f"2. Mapa de Calor ({etapa_nome} - {ano_escolhido})",
    f"3. Resumo dos Perfis ({etapa_nome} - {ano_escolhido})",
    f"4. Gráfico de Dispersão ({etapa_nome} - {ano_escolhido})",
    f"5. Consulta de Estudantes ({etapa_nome} - {ano_escolhido})"
])

# --- ABA 1: Decisão do K ---
with aba_decisao:
    st.subheader(f"Validação Estatística - {etapa_nome} de {ano_escolhido}")
    col_g, col_t = st.columns([1.2, 1])
    with col_g:
        st.markdown("**Método do Cotovelo (WCSS)**")
        fig_cotovelo, ax_cotovelo = plt.subplots(figsize=(6, 4))
        ax_cotovelo.plot(range(1, 11), wcss, marker='o', color='steelblue')
        ax_cotovelo.axvline(x=k_escolhido, color='crimson', linestyle='--', label=f'k Selecionado ({k_escolhido})')
        ax_cotovelo.set_xlabel('Número de Grupos (k)')
        ax_cotovelo.set_ylabel('Inércia Amostral')
        ax_cotovelo.legend()
        st.pyplot(fig_cotovelo)
    with col_t:
        st.markdown("**Comparativo de Índices**")
        st.dataframe(tabela_validacao, use_container_width=True, hide_index=True)

# --- ABA 2: Mapa de Calor (Matriz Completa) ---
with aba_heatmap:
    st.subheader(f"Matriz de Correlação Completa - {etapa_nome} ({ano_escolhido})")

    colunas_interesse = [
        'Transporte_1', 'BF_1', 'M_Leitura', 'M_Registro', 'M_Gênero', 'M_Coecoesão',
        'M_Português', 'F_Português', 'M_Matemática', 'F_Matemática', 'M_Ciências', 'F_Ciências',
        'M_História', 'F_História', 'M_Geografia', 'F_Geografia', 'M_Arte', 'F_Arte',
        'M_EdFísica', 'F_EdFísica', 'M_Inglês', 'F_Inglês', 'Média Global', 'Faltas Médias'
    ]
    cols_heatmap = [col for col in colunas_interesse if col in dados_filtrados.columns]

    matriz_corr = dados_filtrados[cols_heatmap].corr()

    # Plot da matriz completa (sem máscara para visualização total)
    fig_heat, ax_heat = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax_heat
    )
    ax_heat.set_title(f'Mapa de Calor Completo - {etapa_nome} ({ano_escolhido})', fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig_heat)

# --- ABA 3: Resumo dos Perfis ---
with aba_perfis:
    st.subheader(f"Médias das Variáveis por Perfil (k = {k_escolhido})")
    st.dataframe(analise_perfis.style.format("{:.2f}"), use_container_width=True)

# --- ABA 4: Gráfico de Dispersão Interativo (Plotly) ---
with aba_dispersao:
    st.subheader(f"Dispersão dos Estudantes por Perfil - {etapa_nome} ({ano_escolhido})")

    dados_filtrados['Grupo_Rotulo'] = 'Grupo ' + dados_filtrados['Grupo'].astype(str)

    fig_disp = px.scatter(
        dados_filtrados,
        x='Média Global',
        y='Faltas Médias',
        color='Grupo_Rotulo',
        hover_data=['Matrícula'],
        title=f'Perfis de Alunos - {etapa_nome} ({ano_escolhido})',
        symbol='Grupo_Rotulo'
    )

    fig_disp.update_layout(
        xaxis_title="Média Global",
        yaxis_title="Faltas Médias",
        legend_title="Grupos"
    )

    st.plotly_chart(fig_disp, use_container_width=True)

# --- ABA 5: Consulta de Alunos ---
with aba_estudantes:
    st.subheader(f"Lista de Estudantes - {etapa_nome} ({ano_escolhido})")
    grupo_filtro = st.selectbox("Selecione o Grupo:", options=sorted(dados_filtrados['Grupo'].unique()))
    alunos_filtrados = dados_filtrados[dados_filtrados['Grupo'] == grupo_filtro]

    cols_mostrar = [c for c in ['Matrícula', 'Média Global', 'Faltas Médias', 'Grupo'] if c in dados_filtrados.columns]
    st.write(f"Total de **{len(alunos_filtrados)}** estudantes alocados no Grupo {grupo_filtro}:")
    st.dataframe(alunos_filtrados[cols_mostrar], use_container_width=True)
