import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# IMPORTAÇÃO DAS SUAS FUNÇÕES PERSONALIZADAS DO ML_MODELS.PY:
from ml_models import calcular_metricas_k, sugerir_k_recomendado, rotular_perfis



st.set_page_config(
    page_title="Painel Educacional de Agrupamento", layout="wide"
)
st.title("Sistema Adaptativo de Agrupamento de Estudantes (K-Means)")

# ---------------------------------------------------------
# 1. CARREGAMENTO DINÂMICO DE DADOS
# ---------------------------------------------------------
st.sidebar.header("1. Fonte dos Dados")

arquivo_carregado = st.sidebar.file_uploader(
    "Envie a planilha da escola (.xlsx ou .csv):", type=["xlsx", "csv"]
)


@st.cache_data
def carregar_dados(uploaded_file):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Fallback para a base padrão
            caminho_padrao = "Dados Fundamental 2 Limpos 2023, 2024 e 2025.xlsx"
            df = pd.read_excel(caminho_padrao)
        return df.dropna(how="all")
    except Exception as e:
        st.error(
            f"Erro ao ler a base de dados: {e}. Verifique o formato do arquivo."
        )
        return None


df_raw = carregar_dados(arquivo_carregado)

if df_raw is not None and not df_raw.empty:

    # ---------------------------------------------------------
    # 2. CONFIGURAÇÃO DINÂMICA DE FILTROS E VARIÁVEIS
    # ---------------------------------------------------------
    st.sidebar.header("2. Filtros e Configurações")

    # Detecção automática de colunas
    todas_colunas = df_raw.columns.tolist()
    colunas_numericas = df_raw.select_dtypes(include=["number"]).columns.tolist()

    # Seleção de Filtros (Ano / Etapa / Série)
    coluna_ano = st.sidebar.selectbox(
        "Coluna de Ano/Temporal (opcional):",
        options=["Nenhum"] + todas_colunas,
        index=todas_colunas.index("Ano") + 1 if "Ano" in todas_colunas else 0,
    )

    coluna_etapa = st.sidebar.selectbox(
        "Coluna de Etapa/Série (opcional):",
        options=["Nenhum"] + todas_colunas,
        index=todas_colunas.index("Etapa_1") + 1
        if "Etapa_1" in todas_colunas
        else 0,
    )

    # Filtragem dos Dados
    df_filtrado = df_raw.copy()

    if coluna_ano != "Nenhum":
        anos_unicos = sorted(df_raw[coluna_ano].dropna().unique())
        ano_sel = st.sidebar.selectbox("Selecione o Ano:", anos_unicos)
        df_filtrado = df_filtrado[df_filtrado[coluna_ano] == ano_sel]

    if coluna_etapa != "Nenhum":
        etapas_unicas = sorted(df_filtrado[coluna_etapa].dropna().unique())
        etapa_sel = st.sidebar.selectbox("Selecione a Etapa (Série):", etapas_unicas)
        df_filtrado = df_filtrado[df_filtrado[coluna_etapa] == etapa_sel]

    # Identificador do Aluno
    coluna_id = st.sidebar.selectbox(
        "Coluna de Identificação do Aluno (ex: Matrícula):",
        options=todas_colunas,
        index=todas_colunas.index("Matrícula")
        if "Matrícula" in todas_colunas
        else 0,
    )

    # Seleção das Features para Clusterização
    colunas_excluir_padrao = [
        coluna_id,
        coluna_ano,
        coluna_etapa,
        "Média Global",
        "Faltas Médias",
    ]
    features_sugeridas = [
        c for c in colunas_numericas if c not in colunas_excluir_padrao
    ]

    features_selecionadas = st.sidebar.multiselect(
        "Selecione as variáveis para o agrupamento:",
        options=colunas_numericas,
        default=[],
        placeholder="Escolha as variáveis"  # <-- Altera o texto exibido no campo vazio
        if len(features_sugeridas) > 0
        else colunas_numericas,
    )

    k_escolhido = st.sidebar.slider(
        "Quantidade de grupos (k):",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
    )

    # ---------------------------------------------------------
    # 3. VALIDAÇÕES E PROCESSAMENTO DOS DADOS
    # ---------------------------------------------------------
    if len(df_filtrado) < k_escolhido:
        st.warning(
            f"A quantidade de alunos filtrados ({len(df_filtrado)}) é menor que o número de clusters 'k' ({k_escolhido}). Reduza o valor de k ou ajuste os filtros."
        )
    elif len(features_selecionadas) < 2:
        st.warning(
            "Selecione pelo menos 2 variáveis numéricas na barra lateral para realizar o agrupamento."
        )
    else:
        try:
            # Imputação e Padronização
            imputer = KNNImputer(n_neighbors=min(5, len(df_filtrado)))
            df_imputado = df_filtrado.copy()
            df_imputado[features_selecionadas] = imputer.fit_transform(
                df_filtrado[features_selecionadas]
            )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_imputado[features_selecionadas])

            # Cálculo prévio do Cotovelo e Validação
            max_k_possible = min(10, len(df_filtrado) - 1)
            wcss = []
            for i in range(1, max_k_possible + 1):
                km = KMeans(
                    n_clusters=i, init="k-means++", n_init=10, random_state=42
                )
                km.fit(X_scaled)
                wcss.append(km.inertia_)

            metricas = []
            for k_val in range(2, min(6, len(df_filtrado))):
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
            tabela_validacao = pd.DataFrame(metricas)

            # Modelo Final com k escolhido
            kmeans = KMeans(
                n_clusters=k_escolhido,
                init="k-means++",
                n_init=25,
                max_iter=500,
                tol=0.00005,
                random_state=42,
            )

            labels = kmeans.fit_predict(X_scaled)

            # 1. Atribui o grupo aos estudantes
            df_imputado["Grupo"] = labels

            # 2. Mapeia os rótulos pedagógicos (Avançado, Regular, Necessita de Reforço)
            mapeamento_rotulos = rotular_perfis(
                df_imputado,
                nome_col_grupo="Grupo",
                col_media="Média Global",
                col_faltas="Faltas Médias",
            )

            # 3. Cria as colunas explicativas de grupo e perfil
            df_imputado["Perfil_Cluster"] = df_imputado["Grupo"].map(mapeamento_rotulos)
            df_imputado["Grupo_Rotulo"] = (
                "Grupo "
                + df_imputado["Grupo"].astype(str)
                + " ("
                + df_imputado["Perfil_Cluster"]
                + ")"
            )

            # 4. Tabela de Perfis
            dict_agg = {col: "mean" for col in features_selecionadas}
            if "Média Global" in df_imputado.columns:
                dict_agg["Média Global"] = "mean"
            if "Faltas Médias" in df_imputado.columns:
                dict_agg["Faltas Médias"] = "mean"
            dict_agg[coluna_id] = "count"

            analise_perfis = (
                df_imputado.groupby(["Grupo", "Perfil_Cluster"]).agg(dict_agg).reset_index()
            )
            analise_perfis = analise_perfis.rename(
                columns={
                    "Grupo": "grupo_cluster",
                    "Perfil_Cluster": "perfil_cluster",
                    coluna_id: "Qtd_Alunos",
                }
            )
            analise_perfis["rotulo_grupo"] = "Grupo " + analise_perfis[
                "grupo_cluster"
            ].astype(str)

            

            # ---------------------------------------------------------
            # 4. ESTRUTURA EM ABAS
            # ---------------------------------------------------------
            aba_decisao, aba_heatmap, aba_perfis, aba_dispersao, aba_estudantes = (
                st.tabs(
                    [
                        "1. Decisão do k",
                        "2. Mapa de Calor",
                        "3. Resumo dos Perfis",
                        "4. Gráfico de Dispersão",
                        "5. Consulta & Exportação",
                    ]
                )
            )

            # Aba 1: Métricas & Cotovelo

            
            
            
            with aba_decisao:
                st.subheader("Validação Estatística para Escolha da quantidade de grupos (k)")


                tabela_validacao = calcular_metricas_k(X_scaled)
                k_sugerido = sugerir_k_recomendado(tabela_validacao)

                st.info(
                    f"💡 **Recomendação Algorítmica:** Com base na combinação dos índices "
                    f"(Silhouette, Davies-Bouldin e Calinski-Harabasz), o número ideal sugerido é **k = {k_sugerido}**."
                    f"Selecione no **cursor ao lado** a quantidade de grupos de interesse"
                )

                st.markdown("---")
                
                #Gráfico do Método do Cotovelo
                st.markdown("**Método do Cotovelo (WCSS)**")

                # Há 3 colunas, mas a imagem ocupará apenas na coluna do meio, chamada "col_meio"
                col_esq, col_meio, col_dir = st.columns([1, 2, 1])

                with col_meio:
                    fig_cotovelo, ax_cotovelo = plt.subplots(figsize=(6, 4))
                    ax_cotovelo.plot(
                        range(1, len(wcss) + 1),
                        wcss,
                        marker="o",
                        color="steelblue",
                    )
                    ax_cotovelo.axvline(
                        x=k_escolhido,
                        color="crimson",
                        linestyle="--",
                        label=f"k Selecionado ({k_escolhido})",
                    )
                    ax_cotovelo.set_xlabel("Número de Grupos (k)")
                    ax_cotovelo.set_ylabel("Inércia Amostral")
                    ax_cotovelo.legend()
                    st.pyplot(fig_cotovelo)

                st.markdown("---")  # Linha divisória para separar as regiões

                #Tabela comparativa com os índices
                st.markdown("**Comparativo de Índices**")


                st.dataframe(
                    tabela_validacao,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "k": st.column_config.NumberColumn("k", alignment="center"),
                        "Silhouette (↑)": st.column_config.NumberColumn(
                            "Silhouette (↑)", format="%.3f", alignment="center"
                        ),
                        "Davies-Bouldin (↓)": st.column_config.NumberColumn(
                            "Davies-Bouldin (↓)", format="%.3f", alignment="center"
                        ),
                        "Calinski-Harabasz (↑)": st.column_config.NumberColumn(
                            "Calinski-Harabasz (↑)", format="%.3f", alignment="center"
                        ),
                    },
                )
             
                

            # Aba 2: Mapa de Calor
            with aba_heatmap:
                st.subheader("Matriz de Correlação Completa das Variáveis")
                matriz_corr = df_imputado[features_selecionadas].corr()
                fig_heat, ax_heat = plt.subplots(
                    figsize=(min(16, len(features_selecionadas)), 10)
                )
                sns.heatmap(
                    matriz_corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm_r",
                    vmin=-1,
                    vmax=1,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    ax=ax_heat,
                )
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig_heat)

            # Aba 3: Perfis
            with aba_perfis:
                st.subheader("Resumo Consolidado dos Perfis")

                # Monta a agregação com a média de todas as variáveis
                dict_agg = {col: "mean" for col in features_selecionadas}
                if "Média Global" in df_imputado.columns:
                    dict_agg["Média Global"] = "mean"
                if "Faltas Médias" in df_imputado.columns:
                    dict_agg["Faltas Médias"] = "mean"
                dict_agg[coluna_id] = "count"

                tabela_perfis = df_imputado.groupby(
                    ["Grupo", "Perfil_Cluster"]
                ).agg(dict_agg).reset_index()
                tabela_perfis = tabela_perfis.rename(
                    columns={
                        "Grupo": "grupo_cluster",
                        "Perfil_Cluster": "perfil_cluster",
                        coluna_id: "Qtd_Alunos",
                    }
                )

                tabela_perfis["rotulo_grupo"] = "Grupo " + tabela_perfis[
                    "grupo_cluster"
                ].astype(str)

                # Reordena as colunas para ficar idêntico ao layout da sua colega
                colunas_inicio = ["grupo_cluster", "perfil_cluster", "rotulo_grupo", "Qtd_Alunos"]
                outras_colunas = [c for c in tabela_perfis.columns if c not in colunas_inicio]
                tabela_perfis = tabela_perfis[colunas_inicio + outras_colunas]

                # Exibe a tabela estilizada com 3/4 casas decimais
                st.dataframe(
                    tabela_perfis.style.format(
                        {col: "{:.3f}" for col in outras_colunas}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            

            
#            with aba_perfis:
 #               st.subheader(f"Médias das variáveis por grupo (k = {k_escolhido})")
  #              st.dataframe(
   #                 analise_perfis.style.format("{:.2f}"),
    #                use_container_width=True,
     #           )

            # Aba 4: Dispersão
            with aba_dispersao:
                st.subheader("Gráfico de Dispersão dos Estudantes")
                eixo_x = st.selectbox(
                    "Selecione a variável para o Eixo X:",
                    options=colunas_numericas,
                    index=colunas_numericas.index("Média Global")
                    if "Média Global" in colunas_numericas
                    else 0,
                )
                eixo_y = st.selectbox(
                    "Selecione a variável para o Eixo Y:",
                    options=colunas_numericas,
                    index=colunas_numericas.index("Faltas Médias")
                    if "Faltas Médias" in colunas_numericas
                    else min(1, len(colunas_numericas) - 1),
                )

                fig_disp = px.scatter(
                    df_imputado,
                    x=eixo_x,
                    y=eixo_y,
                    color="Grupo_Rotulo",
                    hover_data=[coluna_id],
                    title=f"Perfis de Alunos ({eixo_x} vs {eixo_y})",
                    symbol="Grupo_Rotulo",
                )
                st.plotly_chart(fig_disp, use_container_width=True)

            # Aba 5: Consulta e Exportação
            with aba_estudantes:
                st.subheader("Lista de Estudantes e Exportação")
                grupo_filtro = st.selectbox(
                    "Selecione o Grupo:",
                    options=sorted(df_imputado["Grupo"].unique()),
                )
                alunos_filtrados = df_imputado[
                    df_imputado["Grupo"] == grupo_filtro
                ]

                cols_mostrar = [
                    c
                    for c in [
                        coluna_id,
                        "Média Global",
                        "Faltas Médias",
                        "Grupo",
                    ]
                    if c in df_imputado.columns
                ]
                st.write(
                    f"Total de **{len(alunos_filtrados)}** estudantes no Grupo {grupo_filtro}:"
                )
                st.dataframe(
                    alunos_filtrados[cols_mostrar], use_container_width=True
                )

                # Botão para exportar resultados
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_imputado.to_excel(
                        writer, sheet_name="Alunos_Clusterizados", index=False
                    )
                    analise_perfis.to_excel(
                        writer, sheet_name="Resumo_Perfis"
                    )

                st.download_button(
                    label="📥 Baixar Planilha com Grupos Atribuídos (.xlsx)",
                    data=buffer.getvalue(),
                    file_name="Resultado_Clusterizacao_Estudantes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"Erro no processamento dos dados ou do modelo: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo válido para começar.")
