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

def expand_column_terms(column_name):
    """Usa IA para generar sinónimos y equivalencias de una columna detectada en el Excel."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en archivística y catalogación según ISDF."},
            {"role": "user", "content": f"Genera sinónimos y términos equivalentes para '{column_name}' en el contexto de archivos y catalogación."}
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

def extract_column_definitions(pdf_text, column_names):
    """Extrae definiciones de las columnas desde el texto del PDF usando IA."""
    column_definitions = {}
    for col in column_names:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en archivística y catalogación según ISDF."},
                {"role": "user", "content": f"Extrae la definición más relevante del término '{col}' según el siguiente documento:\n{pdf_text}"}
            ],
            max_tokens=200
        )
        column_definitions[col] = response.choices[0].message.content.strip()
    return column_definitions

# Extraer definiciones específicas de columnas desde el PDF
atom_template = pd.read_csv("Example_information_objects_isad-2.8.csv")
column_definitions = extract_column_definitions(ISDF_FULL_TEXT, atom_template.columns)

def generate_column_embeddings(atom_columns):
    """Genera embeddings de las columnas usando sus definiciones extraídas del ISDF."""
    enriched_texts = {col: f"{col} - {column_definitions.get(col, '')}" for col in atom_columns}
    return {col: get_embedding(text) for col, text in enriched_texts.items()}

# Precalcular embeddings de columnas de referencia
atom_embeddings = generate_column_embeddings(atom_template.columns)

def match_exact_or_approximate(column_name):
    """Intenta encontrar una coincidencia exacta o aproximada utilizando términos expandidos."""
    if column_name not in TERM_EQUIVALENTS:
        TERM_EQUIVALENTS[column_name] = expand_column_terms(column_name)
    for key, values in TERM_EQUIVALENTS.items():
        if any(clean_text(column_name) == clean_text(value) for value in values):
            return key
    return None

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
    exact_match = match_exact_or_approximate(column_name)
    if exact_match:
        return exact_match
    
    cleaned_column_name = clean_text(column_name)
    combined_text = cleaned_column_name + " " + column_definitions.get(column_name, '')
    column_embedding = get_embedding(combined_text)
    
    similarity_scores = {col: cosine_similarity([column_embedding], [emb])[0][0] for col, emb in atom_embeddings.items()}
    
    # Ordenar y seleccionar las 5 mejores coincidencias
    top_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.write(f"Top 5 similitudes para {column_name}:", top_matches)
    
    # Evitar que columnas de baja prioridad se seleccionen si hay mejores opciones
    for match in top_matches:
        if match[0] not in LOW_PRIORITY_COLUMNS:
            return match[0]
    
    return top_matches[0][0] if top_matches[0][1] > 0.80 else None

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    # Leer el archivo Excel
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    output_df = pd.DataFrame(columns=atom_template.columns)
    
    # Intentar mapear automáticamente las columnas detectadas en el Excel cargado
    column_mapping = {}
    combined_columns = {}
    
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
    
    st.write("Mapa de columnas detectadas y ajustadas con mayor precisión usando embeddings y términos expandidos:")
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