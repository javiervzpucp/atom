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

# Obtener embeddings
def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

# Agrupar columnas ISAD por tema
ISAD_GROUPS = {
    "date": ["dateCreated", "dateIssued", "dateModified", "recordCreationDate", "eventDate"],
    "identifier": ["identifier", "recordIdentifier", "institutionIdentifier"],
    "description": ["title", "scopeAndContent", "archivalHistory", "levelOfDescription"],
    "relations": ["relatedUnitsOfDescription", "scriptOfDescription"],
}

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
    
    if len(values) == 0:
        return "unknown"
    
    date_count = sum(any(re.match(pattern, str(v)) for pattern in date_patterns) for v in values)
    total_values = len(values)
    
    if date_count / total_values > 0.7:
        return "date"
    return "mixed"

# Asignación basada en embeddings y similitud coseno
def find_best_match(column_name, column_type):
    """Encuentra la mejor coincidencia en el template usando embeddings y distancia coseno, dentro del grupo adecuado."""
    if column_type in ISAD_GROUPS:
        relevant_columns = ISAD_GROUPS[column_type]
    else:
        relevant_columns = sum(ISAD_GROUPS.values(), [])  # Todas las columnas
    
    column_embedding = get_embedding(column_name)
    reference_embeddings = {col: get_embedding(col) for col in relevant_columns}
    
    similarities = {
        col: cosine_similarity([column_embedding], [emb])[0][0]
        for col, emb in reference_embeddings.items()
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
    
    # Crear nuevo DataFrame con las columnas mapeadas
    converted_df = pd.DataFrame()
    for original_col, mapped_col in column_contexts.items():
        converted_df[mapped_col] = df[original_col]
    
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