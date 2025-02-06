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

# Clasificación automática de tipo de datos basado en valores

def classify_column_type(values):
    """Determina el tipo de datos basado en los valores de la columna."""
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',  # Formato: DD/MM/YYYY
        r'\d{4}-\d{2}-\d{2}',  # Formato: YYYY-MM-DD
        r'\d{2}-\d{2}-\d{4}',  # Formato: DD-MM-YYYY
    ]
    numeric_pattern = r'^[0-9]+$'
    
    date_count = sum(any(re.match(pattern, v) for pattern in date_patterns) for v in values)
    unique_values = set(values)
    numeric_count = sum(re.match(numeric_pattern, v) is not None for v in values)
    
    total_values = len(values)
    
    if date_count / total_values > 0.7:
        return "date"
    elif numeric_count / total_values > 0.7 and len(unique_values) == total_values:
        return "identifier"
    return "mixed"

# Obtener embeddings

def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

# Mejor coincidencia con columnas del template

def find_best_date_column(reference_columns):
    """Busca en las columnas de referencia aquellas que contengan 'date' en su nombre."""
    for col in reference_columns:
        if "date" in col.lower():
            return col
    return None

# Extraer información semántica de las columnas del Excel cargado

def extract_column_context(df, reference_columns):
    """Extrae información semántica de las columnas del Excel basado en nombres y valores."""
    column_contexts = {}
    for column in df.columns:
        expanded_column = clean_text(column)
        values_sample = df[column].dropna().astype(str).tolist()[:5]
        column_type = classify_column_type(values_sample)
        
        # Si la columna es de fecha, buscar coincidencias con columnas de referencia que contengan 'date'
        if column_type == "date":
            best_match = find_best_date_column(reference_columns)
            if best_match:
                column_contexts[column] = best_match
                continue
        
        expanded_terms = expand_column_terms(expanded_column, values_sample)
        column_text = f"{expanded_column} {' '.join(expanded_terms)} Tipo: {column_type} {' '.join(values_sample)}"
        column_contexts[column] = get_embedding(column_text)
    return column_contexts

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    # Obtener nombres de columnas del template ISAD 2.8
    reference_columns = ["identifier", "dateCreated", "dateIssued", "dateModified", "recordCreationDate"]
    column_contexts = extract_column_context(df, reference_columns)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Datos Convertidos")
    output.seek(0)
    
    st.download_button(
        label="Descargar archivo en formato ISAD 2.8",
        data=output,
        file_name="archivados_isad_2.8.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )