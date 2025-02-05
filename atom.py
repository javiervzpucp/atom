# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:45:36 2025

@author: jveraz
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from openai import OpenAI
import numpy as np
import re
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

# Configurar la aplicación Streamlit
st.title("Archivador Inteligente de Documentos Antiguos")

# Inicializar cliente de OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Extraer texto del documento ISDF
ISDF_DOC_PATH = "CBPS_2007_Guidelines_ISDF_First-edition_SP.docx"

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text[:2000]  # Limitar el tamaño del texto extraído

# Cargar contenido del documento ISDF para usar en embeddings
ISDF_FULL_TEXT = extract_text_from_docx(ISDF_DOC_PATH)

def clean_text(text):
    """Normaliza el texto eliminando caracteres especiales y pasando a minúsculas."""
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower().strip())

@st.cache_data
def get_embeddings_batch(texts):
    """Obtiene los embeddings de OpenAI en batch para mejorar el rendimiento."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(item.embedding) for item in response.data]

@st.cache_data
def precalculate_atom_embeddings(atom_columns):
    """Precalcula y almacena los embeddings de las columnas del formato ISAD 2.8."""
    return dict(zip(atom_columns, get_embeddings_batch(atom_columns)))

def find_best_match(column_name, column_values, atom_embeddings):
    """Encuentra la mejor coincidencia usando embeddings de OpenAI considerando los valores de la columna y la norma ISDF."""
    cleaned_column_name = clean_text(column_name)
    combined_text = cleaned_column_name + " " + " ".join(map(str, column_values[:3]))  # Reducimos el tamaño del texto analizado
    column_embedding = get_embeddings_batch([combined_text])[0]
    
    similarity_scores = {col: cosine_similarity([column_embedding], [emb])[0][0] for col, emb in atom_embeddings.items()}
    
    # Ordenar y seleccionar las 5 mejores coincidencias
    top_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.write(f"Top 5 similitudes para {column_name}:", top_matches)
    
    return top_matches[0][0] if top_matches[0][1] > 0.75 else None

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    # Leer el archivo Excel
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    # Cargar el formato ISAD 2.8 desde el CSV actualizado
    atom_template = pd.read_csv("Example_information_objects_isad-2.8.csv")
    output_df = pd.DataFrame(columns=atom_template.columns)
    
    # Precalcular embeddings para todas las columnas de ISAD 2.8
    atom_embeddings = precalculate_atom_embeddings(atom_template.columns)
    
    # Intentar mapear automáticamente las columnas detectadas en el Excel cargado
    column_mapping = {}
    for column in df.columns:
        best_match = find_best_match(column, df[column].dropna().astype(str).tolist(), atom_embeddings)
        if best_match:
            output_df[best_match] = df[column].fillna("N/A")
            column_mapping[column] = best_match
    
    st.write("Mapa de columnas detectadas y ajustadas con mayor precisión usando embeddings y datos de muestra con ISDF:")
    st.write(column_mapping)
    
    # Convertir a Excel para descarga
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name="Datos Convertidos")
    output.seek(0)
    
    st.download_button(
        label="Descargar archivo en formato ISAD 2.8",
        data=output,
        file_name="archivados_isad_2.8.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )