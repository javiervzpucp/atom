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
    return text

# Cargar contenido del documento ISDF para usar en embeddings
ISDF_FULL_TEXT = extract_text_from_docx(ISDF_DOC_PATH)

def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def find_best_match(column_name, column_values, atom_columns):
    """Encuentra la mejor coincidencia usando embeddings de OpenAI considerando los valores de la columna y la norma ISDF."""
    combined_text = column_name + " " + " ".join(map(str, column_values[:5])) + " " + ISDF_FULL_TEXT[:2000]
    column_embedding = get_embedding(combined_text)
    
    best_match = None
    best_similarity = -1
    atom_embeddings = {col: get_embedding(col) for col in atom_columns}  # Precalcular embeddings
    
    similarity_scores = {}
    for atom_field, atom_embedding in atom_embeddings.items():
        similarity = np.dot(column_embedding, atom_embedding) / (np.linalg.norm(column_embedding) * np.linalg.norm(atom_embedding))
        similarity_scores[atom_field] = similarity
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = atom_field
    
    # Mostrar las similitudes para depuración
    st.write(f"Similitudes calculadas para {column_name}:", similarity_scores)
    
    return best_match if best_similarity > 0.75 else None  # Se mantiene el umbral alto para precisión

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
    
    # Intentar mapear automáticamente las columnas detectadas en el Excel cargado
    column_mapping = {}
    for column in df.columns:
        best_match = find_best_match(column, df[column].dropna().astype(str).tolist(), atom_template.columns)
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
