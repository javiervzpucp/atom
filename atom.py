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

# Obtener embeddings de los keys de ISAD_GROUPS
def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

# Definir claves simplificadas para ISAD_GROUPS
ISAD_GROUPS = {
    "date": ["dateCreated", "dateIssued", "dateModified", "recordCreationDate", "eventDate"],
    "identifier": ["identifier", "recordIdentifier", "institutionIdentifier"],
    "description": ["title", "scopeAndContent", "archivalHistory", "levelOfDescription"],
    "relations": ["relatedUnitsOfDescription", "scriptOfDescription"],
}

# Solo trabajamos con los keys simplificados
ISAD_KEYS = list(ISAD_GROUPS.keys())

# Generar embeddings de las claves simplificadas ISAD usando el contenido del documento ISDF
ISAD_KEY_EMBEDDINGS = {key: get_embedding(key) for key in ISAD_KEYS}

# Limpieza de texto con expansión de abreviaturas y palabras pegadas
def expand_text_with_ai(text):
    """Expande abreviaturas, corrige errores tipográficos y separa palabras pegadas usando IA."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en procesamiento de texto y normalización de datos archivísticos."},
            {"role": "user", "content": f"Expande abreviaturas, corrige errores tipográficos y separa palabras pegadas en este texto: {text}"}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def clean_text(text):
    """Normaliza el texto eliminando caracteres especiales y pasando a minúsculas, expandiendo abreviaturas."""
    text = expand_text_with_ai(text)
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower().strip())

# Clasificación automática de tipo de datos basado en valores y nombres
def classify_column_type(column_name, values):
    """Determina el tipo de datos basado en los valores de la columna y su nombre."""
    date_patterns = [
        r'\d{4}$',  # Años exactos (1874, 1902, etc.)
        r'\d{4}-\d{2}-\d{2}',  # Formato YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # Formato DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',  # Formato DD-MM-YYYY
        r's\.XVIII|s\.XVII|s\.XIX',  # Siglos (s.XVIII)
        r'\d{4}-[a-zA-Z]+\.\d{2}',  # Fechas con meses escritos (1909-May.-29)
    ]
    
    # Validación por nombre de columna
    if "fecha" in column_name.lower() or "date" in column_name.lower():
        return "date"
    if "id" in column_name.lower() or "número" in column_name.lower():
        return "identifier"
    
    if len(values) == 0:
        return "unknown"
    
    date_count = sum(any(re.match(pattern, str(v)) for pattern in date_patterns) for v in values)
    total_values = len(values)
    
    if date_count / total_values > 0.7:
        return "date"
    
    return "mixed"

# Asignación basada en embeddings y substrings
def find_best_match(column_name, column_type):
    """Encuentra la mejor coincidencia en ISAD_KEYS usando substrings y embeddings, priorizando categorías adecuadas."""
    relevant_keys = ISAD_KEYS if column_type not in ISAD_GROUPS else [column_type]
    
    # Coincidencia por substring
    for key in relevant_keys:
        if column_type in key.lower():
            return key
    
    # Si no hay coincidencia exacta, usar embeddings
    column_embedding = get_embedding(column_name)
    similarities = {
        key: cosine_similarity([column_embedding], [ISAD_KEY_EMBEDDINGS[key]])[0][0]
        for key in relevant_keys
    }
    
    best_match = max(similarities, key=similarities.get)
    return best_match

# Extraer información semántica de las columnas del Excel cargado
def extract_column_context(df):
    """Extrae información semántica de las columnas del Excel basado en nombres y valores."""
    column_contexts = {}
    for column in df.columns:
        expanded_column = clean_text(column)
        values_sample = df[column].dropna().astype(str).tolist()[:10]
        column_type = classify_column_type(expanded_column, values_sample)
        
        # Buscar mejor coincidencia según tipo de dato y grupo
        best_match = find_best_match(expanded_column, column_type)
        if best_match in ISAD_KEYS:
            column_contexts[column] = best_match
    return column_contexts

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    # Obtener mapeo de columnas
    column_contexts = extract_column_context(df)
    st.write("Mapeo de columnas:")
    st.json(column_contexts)
    
    # Crear nuevo DataFrame con nombres concatenados
    converted_df = pd.DataFrame()
    for original_col, mapped_col in column_contexts.items():
        new_col_name = f"{original_col} ({mapped_col})"
        converted_df[new_col_name] = df[original_col]
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        converted_df.to_excel(writer, index=False, sheet_name="Datos Convertidos")
    output.seek(0)
    
    st.download_button(
        label="Descargar archivo en formato ISAD 2.8",
        data=output,
        file_name="archivados_isad_2.8.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.stop()  # Detener la ejecución después de la descarga
