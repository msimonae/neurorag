import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import google.generativeai as genai
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from collections import Counter

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="NeuroGRAG - Copiloto Clínico", layout="wide")
st.title("🧠 NeuroGRAG: Copiloto Clínico Explicável")
st.markdown("""
Este aplicativo demonstra uma abordagem experimental para apoio ao raciocínio médico em triagem psiquiátrica,
utilizando LLMs, grafos clínicos e Recuperação Aumentada de Geração (RAG).
**Atenção:** Este é um projeto acadêmico e de pesquisa. Não utilize para diagnósticos reais.
""")

# --- Cache para Modelos e Dados ---
@st.cache_resource
def carregar_modelo_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def carregar_dados_e_setup_grafo_chroma():
    try:
        df_transtornos = pd.read_csv("df_transtornos.csv")
    except FileNotFoundError:
        st.error("Arquivo 'df_transtornos.csv' não encontrado. Gere-o a partir do notebook e coloque na raiz do app.")
        return None, None, None

    # 1. Grafo de Conhecimento
    G = nx.Graph()
    diagnosticos_unicos = []
    if not df_transtornos.empty:
        for _, linha in df_transtornos.iterrows():
            diag = linha['diagnóstico']
            if pd.notna(diag) and diag not in diagnosticos_unicos:
                 if isinstance(diag, str) and ('F' in diag or 'ND' in diag):
                    diagnosticos_unicos.append(diag)

            if pd.notna(linha['sintomas_principais']):
                sintomas = [s.strip() for s in str(linha['sintomas_principais']).split(',')]
                for sintoma in sintomas:
                    if pd.notna(diag) and pd.notna(sintoma):
                        G.add_edge(sintoma, diag, tipo='sintoma-diagnostico')

            if pd.notna(linha['tratamento']):
                tratamentos = [t.strip() for t in str(linha['tratamento']).split(',')]
                for trat in tratamentos:
                    if pd.notna(diag) and pd.notna(trat):
                        G.add_edge(diag, trat, tipo='diagnóstico-tratamento')
    else:
        st.warning("DataFrame df_transtornos está vazio. O grafo não será populado.")


    # 2. ChromaDB
    modelo_embed = carregar_modelo_embedding()
    chroma_client = PersistentClient(path='./chroma_saude_mental_st') # Novo path para Streamlit

    # Deletar coleção antiga se existir para evitar conflitos de dimensão (opcional, mas bom para dev)
    try:
        if 'casos_clinicos_st' in [c.name for c in chroma_client.list_collections()]:
            chroma_client.delete_collection(name='casos_clinicos_st')
            st.info("Coleção ChromaDB 'casos_clinicos_st' antiga recriada.")
    except Exception as e:
        st.warning(f"Não foi possível deletar a coleção antiga: {e}")

    colecao = chroma_client.get_or_create_collection(name='casos_clinicos_st')

    if not df_transtornos.empty and 'texto_clinico' in df_transtornos.columns:
        chunks = []
        metadados_list = []
        ids_chunks = []
        tamanho_chunk = 200 # Definido no notebook

        for idx, linha in df_transtornos.iterrows():
            texto = str(linha['texto_clinico']) # Garantir que é string
            palavras = texto.split()
            id_paciente = str(linha['id_paciente'])
            diagnostico = str(linha['diagnóstico'])
            sintomas = str(linha['sintomas_principais'])

            for i_chunk in range(0, len(palavras), tamanho_chunk):
                trecho = ' '.join(palavras[i_chunk:i_chunk + tamanho_chunk])
                chunk_id = f'{id_paciente}_chunk_{i_chunk // tamanho_chunk}'
                chunks.append(trecho)
                ids_chunks.append(chunk_id)
                metadados_list.append({
                    'id_paciente': id_paciente,
                    'diagnóstico': diagnostico,
                    'sintomas': sintomas,
                    'indice_chunk': i_chunk // tamanho_chunk
                })

        if chunks: # Apenas se houver chunks para processar
            embeddings = modelo_embed.encode(chunks)
            colecao.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadados_list,
                ids=ids_chunks
            )
            st.success(f'{len(chunks)} chunks indexados no ChromaDB.')
        else:
            st.warning("Nenhum chunk gerado para indexação no ChromaDB. Verifique os dados de 'texto_clinico'.")

    else:
        st.warning("Coluna 'texto_clinico' não encontrada ou df_transtornos vazio. ChromaDB não será populado.")

    return df_transtornos, G, colecao

# --- Carregamento Inicial ---
modelo_embedding = carregar_modelo_embedding()
df_transtornos, G_conhecimento, colecao_chroma = carregar_dados_e_setup_grafo_chroma()

# --- Configuração da API Key do Gemini ---
# Para Streamlit Cloud, use st.secrets
# Para desenvolvimento local, use st.text_input
google_api_key = st.sidebar.text_input("Cole sua GOOGLE_API_KEY do Gemini aqui:", type="password")

modelo_gemini = None
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        modelo_gemini = genai.GenerativeModel('gemini-1.5-pro') # ou gemini-pro
        st.sidebar.success("API Key do Gemini configurada com sucesso!")
    except Exception as e:
        st.sidebar.error(f"Erro ao configurar a API Key: {e}")
else:
    st.sidebar.warning("Por favor, insira sua API Key do Gemini para continuar.")

# --- Interface do Usuário ---
st.header("Entrada do Caso Clínico")
relato_paciente_input = st.text_area("Descreva o relato do paciente (sintomas, histórico, etc.):", height=150)
sintomas_chave_input = st.text_input("Informe os sintomas-chave detectados (separados por vírgula):",
                                     help="Ex: tristeza profunda, fadiga, isolamento social")

if st.button("Analisar Caso Clínico") and modelo_gemini and G_conhecimento and colecao_chroma:
    if not relato_paciente_input.strip():
        st.error("Por favor, preencha o relato do paciente.")
    elif not sintomas_chave_input.strip():
        st.error("Por favor, informe os sintomas-chave.")
    else:
        with st.spinner("Analisando o caso... Este processo pode levar alguns minutos."):
            sintomas_detectados_lista = [s.strip().lower() for s in sintomas_chave_input.split(',')] # Convertido para minúsculas

            # 1. Consulta ao Grafo de Conhecimento
            diagnosticos_possiveis_grafo = set()
            if G_conhecimento is not None:
                for sintoma_g in sintomas_detectados_lista:
                    # Tentar encontrar o sintoma no grafo (considerando variações)
                    nos_candidatos = [n for n in G_conhecimento.nodes() if isinstance(n, str) and sintoma_g in n.lower()]
                    for no_candidato in nos_candidatos:
                        if G_conhecimento.has_node(no_candidato):
                            conexoes = list(G_conhecimento.neighbors(no_candidato))
                            diagnosticos_possiveis_grafo.update([d for d in conexoes if isinstance(d, str) and ('F' in d or 'ND' in d)])

            st.subheader("Resultados da Análise:")
            with st.expander("Diagnósticos Sugeridos pelo Grafo de Conhecimento", expanded=False):
                if diagnosticos_possiveis_grafo:
                    for diag_g in diagnosticos_possiveis_grafo:
                        st.write(f"- {diag_g}")
                else:
                    st.write("Nenhum diagnóstico diretamente conectado aos sintomas-chave foi encontrado no grafo.")

            # 2. Consulta ao Banco Vetorial (RAG)
            rag_resultados_formatados = ["Nenhum caso semelhante encontrado no banco vetorial."]
            if colecao_chroma is not None and relato_paciente_input:
                embedding_query = modelo_embedding.encode([relato_paciente_input])
                try:
                    resultados_rag = colecao_chroma.query(
                        query_embeddings=embedding_query.tolist(),
                        n_results=3
                    )

                    docs_recuperados = []
                    if resultados_rag and resultados_rag['documents'] and resultados_rag['documents'][0]:
                        for i, doc_rag in enumerate(resultados_rag['documents'][0]):
                            meta_rag = resultados_rag['metadatas'][0][i]
                            docs_recuperados.append(
                                f"  Caso {i+1} (ID: {meta_rag.get('id_paciente', 'N/A')}, Diagnóstico: {meta_rag.get('diagnóstico', 'N/A')}):\n  Trecho: \"{doc_rag}\"\n  Sintomas Registrados: {meta_rag.get('sintomas', 'N/A')}"
                            )
                        rag_resultados_formatados = docs_recuperados

                except Exception as e:
                    st.error(f"Erro ao consultar o ChromaDB: {e}")

            with st.expander("Casos Clínicos Semelhantes (RAG)", expanded=False):
                for res_rag in rag_resultados_formatados:
                    st.markdown(res_rag)

            # 3. Geração da Resposta com LLM (Gemini)
            prompt_final_template = f'''
              Você é um assistente clínico especializado em saúde mental, treinado para analisar sintomas, sugerir diagnósticos e indicar condutas terapêuticas com base em conhecimento estruturado.

              Relato do Paciente:
              {relato_paciente_input}

              Sintomas-Chave Identificados:
              {', '.join(sintomas_detectados_lista)}

              Diagnósticos Possíveis Sugeridos pelo Grafo de Conhecimento (conexões diretas com sintomas-chave):
              {chr(10).join(diagnosticos_possiveis_grafo) if diagnosticos_possiveis_grafo else "Nenhum diagnóstico diretamente conectado encontrado no grafo."}

              Casos Clínicos Semelhantes Recuperados do Banco Vetorial (para contextualização):
              {chr(10).join(rag_resultados_formatados)}

              Com base NO RELATO DO PACIENTE, nos SINTOMAS-CHAVE IDENTIFICADOS, e considerando o CONTEXTO dos diagnósticos do grafo e dos casos recuperados, forneça uma análise clínica:

              1.  **Diagnóstico Mais Provável:** Indique o CID-10 e o nome do transtorno que você considera mais compatível com o relato completo do paciente.
              2.  **Justificativa Clínica:** Explique sua escolha diagnóstica, correlacionando os sintomas específicos do relato do paciente com os critérios diagnósticos. Mencione se o histórico familiar (se houver no relato) ou outros fatores são relevantes.
              3.  **Conduta Inicial Sugerida:** Recomende uma conduta terapêutica inicial (ex: tipo de medicação geral se aplicável, tipo de psicoterapia, necessidade de acompanhamento especializado).
              4.  **Avaliação de Risco e Encaminhamento:** Avalie brevemente se há indicadores de risco elevado (ex: ideação suicida, agitação psicomotora severa) e se um encaminhamento urgente para um psiquiatra ou serviço de emergência é necessário.

              Responda com clareza, embasamento e em linguagem clínica adequada. Foque no caso atual do paciente.
            '''

            st.info("Gerando análise clínica com o modelo Gemini...")
            try:
                resposta_gemini = modelo_gemini.generate_content(prompt_final_template)
                st.subheader("Análise Clínica Detalhada (NeuroGRAG):")
                st.markdown(resposta_gemini.text)
            except Exception as e:
                st.error(f"Erro ao gerar resposta com o Gemini: {e}")
                st.error(f"Prompt enviado ao Gemini (para depuração):\n```\n{prompt_final_template}\n```")

elif not modelo_gemini:
    st.info("Aguardando a inserção da API Key do Gemini para habilitar a análise.")
elif not G_conhecimento or not colecao_chroma:
    st.error("Não foi possível carregar os dados do grafo ou do ChromaDB. Verifique o arquivo 'df_transtornos.csv'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido como parte de um estudo experimental.")
st.sidebar.markdown("Aluno: Marcelo Massashi Simonae")
