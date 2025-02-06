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

# Limpieza de texto con expansión de abreviaturas y palabras pegadas

def expand_text_with_ai(text):
    """Expande abreviaturas y separa palabras pegadas usando IA."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en procesamiento de texto y normalización de datos archivísticos."},
            {"role": "user", "content": f"Expande abreviaturas y corrige palabras pegadas en este texto: {text}"}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def clean_text(text):
    """Normaliza el texto eliminando caracteres especiales y pasando a minúsculas, expandiendo abreviaturas."""
    text = expand_text_with_ai(text)
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower().strip())

# Obtener embeddings

def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

# Expansión dinámica de términos con IA

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

# Extraer información semántica de las columnas del Excel cargado

def extract_column_context(df):
    """Extrae información semántica de las columnas del Excel basado en nombres y valores."""
    column_contexts = {}
    for column in df.columns:
        expanded_column = clean_text(column)
        values_sample = df[column].dropna().astype(str).tolist()[:5]
        expanded_terms = expand_column_terms(expanded_column, values_sample)
        column_text = f"{expanded_column} {' '.join(expanded_terms)} {' '.join(values_sample)}"
        column_contexts[column] = get_embedding(column_text)
    return column_contexts

# Generar resumen de varias columnas fusionadas

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

# Coincidencia de columnas basada en embeddings

def find_best_match(column_name, column_embedding, reference_embeddings):
    """Encuentra la mejor coincidencia basada en embeddings y nombres expandidos."""
    similarity_scores = {col: cosine_similarity([column_embedding], [emb])[0][0] for col, emb in reference_embeddings.items()}
    top_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.write(f"Top 5 similitudes para {column_name}:", top_matches)
    
    for match in top_matches[:3]:
        if match[0] != "archivalHistory":
            return match[0]
    
    return top_matches[0][0] if top_matches[0][1] > 0.80 else None

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    atom_template = pd.read_csv("Example_information_objects_isad-2.8.csv")
    reference_embeddings = {col: get_embedding(col) for col in atom_template.columns}
    
    column_contexts = extract_column_context(df)
    
    output_df = pd.DataFrame(columns=atom_template.columns)
    combined_columns = {}
    column_mapping = {}
    
    for column, embedding in column_contexts.items():
        best_match = find_best_match(column, embedding, reference_embeddings)
        if best_match:
            if best_match in combined_columns:
                combined_columns[best_match].extend(df[column].dropna().astype(str).tolist())
            else:
                combined_columns[best_match] = df[column].dropna().astype(str).tolist()
            column_mapping[column] = best_match
    
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