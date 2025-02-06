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
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# Configurar la aplicación Streamlit
st.title("Archivador Inteligente de Documentos Antiguos")

# Inicializar cliente de OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Extraer texto del documento ISDF desde PDF
ISDF_PDF_PATH = "CBPS_2007_Guidelines_ISDF_First-edition_EN.pdf"

def extract_text_from_pdf(pdf_path):
    """Extrae el texto del PDF para su uso en embeddings."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Cargar contenido del documento ISDF para usar en embeddings
ISDF_FULL_TEXT = extract_text_from_pdf(ISDF_PDF_PATH)

# Expansión dinámica de términos

def expand_column_terms(column_name, sample_values=[]):
    """Usa IA para generar sinónimos y equivalencias de una columna detectada en el Excel."""
    prompt = f"Genera sinónimos y términos equivalentes para '{column_name}' en el contexto de archivos y catalogación."
    if sample_values:
        prompt += f" Considera estos valores de muestra: {', '.join(sample_values[:5])}."
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en archivística y catalogación según ISDF."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip().split(", ")

# Diccionario de términos equivalentes basado en IA
TERM_EQUIVALENTS = {}

# Diccionario de términos de baja prioridad
LOW_PRIORITY_COLUMNS = ["archivalHistory", "relatedUnitsOfDescription", "archivistNote"]

def clean_text(text):
    """Normaliza el texto eliminando caracteres especiales y pasando a minúsculas."""
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower().strip())

def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def summarize_combined_columns(values):
    """Genera un resumen de varias columnas fusionadas usando IA."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en archivística. Resume la siguiente información manteniendo los detalles clave."},
            {"role": "user", "content": f"Resume la siguiente información en una oración coherente: {values}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def find_best_match(column_name, column_values):
    """Encuentra la mejor coincidencia usando match exacto o embeddings de OpenAI."""
    expanded_terms = expand_column_terms(column_name, column_values)
    possible_matches = [column_name] + expanded_terms
    
    # Generar embedding de los términos expandidos
    combined_embedding = np.mean([get_embedding(term) for term in possible_matches], axis=0)
    
    similarity_scores = {col: cosine_similarity([combined_embedding], [emb])[0][0] for col, emb in atom_embeddings.items()}
    
    # Ordenar y seleccionar las 5 mejores coincidencias
    top_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.write(f"Top 5 similitudes para {column_name}:", top_matches)
    
    # Evitar que archivalHistory domine la selección si hay mejores opciones
    for match in top_matches:
        if match[0] not in LOW_PRIORITY_COLUMNS:
            return match[0]
    
    return top_matches[0][0] if top_matches[0][1] > 0.80 else None

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    output_df = pd.DataFrame(columns=atom_template.columns)
    combined_columns = {}
    column_mapping = {}
    
    for column in df.columns:
        best_match = find_best_match(column, df[column].dropna().astype(str).tolist())
        if best_match:
            if best_match in combined_columns:
                combined_columns[best_match].extend(df[column].dropna().astype(str).tolist())
            else:
                combined_columns[best_match] = df[column].dropna().astype(str).tolist()
            column_mapping[column] = best_match
    
    # Generar resúmenes y llenar el DataFrame de salida
    for col, values in combined_columns.items():
        output_df[col] = [summarize_combined_columns(values)]
    
    st.write("Mapa de columnas detectadas y ajustadas:")
    st.write(column_mapping)
    
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