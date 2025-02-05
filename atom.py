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

# Configurar la aplicación Streamlit
st.title("Archivador Inteligente de Documentos Antiguos")

# Inicializar cliente de OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def generate_metadata(description, field):
    """Utiliza OpenAI para generar metadatos específicos según el campo del formato AtoM 2.8."""
    if not description or pd.isna(description):
        return "N/A"
    
    prompt = (
        f"Eres un archivista experto en catalogación de documentos históricos usando el formato AtoM 2.8. "
        f"Genera un contenido preciso para el campo '{field}' con base en la siguiente información del documento: {description}. "
        f"Sigue estrictamente los estándares de AtoM 2.8 para la redacción de metadatos."
    )
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en archivística y catalogación según AtoM 2.8."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def get_embedding(text):
    """Obtiene el embedding de OpenAI para un texto dado."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def find_best_match(column_name, atom_columns):
    """Encuentra la mejor coincidencia usando embeddings de OpenAI y ajusta la similitud para mayor precisión."""
    column_embedding = get_embedding(column_name)
    best_match = None
    best_similarity = -1
    
    for atom_field in atom_columns:
        atom_embedding = get_embedding(atom_field)
        similarity = np.dot(column_embedding, atom_embedding) / (np.linalg.norm(column_embedding) * np.linalg.norm(atom_embedding))
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = atom_field
    
    return best_match if best_similarity > 0.85 else None  # Se aumenta el umbral para mayor precisión

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube un archivo Excel con los documentos", type=["xlsx"])

if uploaded_file:
    # Leer el archivo Excel
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados del archivo:")
    st.dataframe(df)
    
    # Cargar el formato de AtoM 2.8 desde el CSV de referencia
    atom_template = pd.read_csv("Example_authority_records-2.8.csv")
    output_df = pd.DataFrame(columns=atom_template.columns)
    
    # Intentar mapear automáticamente las columnas detectadas en el Excel cargado
    column_mapping = {}
    for column in df.columns:
        best_match = find_best_match(column, atom_template.columns)
        if best_match:
            output_df[best_match] = df[column].fillna("N/A")
            column_mapping[column] = best_match
    
    st.write("Mapa de columnas detectadas y ajustadas con mayor precisión usando embeddings:")
    st.write(column_mapping)
    
    # Generar metadatos adicionales según los campos del formato AtoM 2.8
    st.write("Generando metadatos específicos del formato AtoM 2.8...")
    for field in output_df.columns:
        if output_df[field].isnull().all():
            continue
        output_df[field] = output_df[field].apply(lambda x: generate_metadata(str(x), field) if x != "N/A" else "N/A")
    
    # Convertir a Excel para descarga
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name="Datos Convertidos")
    output.seek(0)
    
    st.download_button(
        label="Descargar archivo en formato AtoM 2.8",
        data=output,
        file_name="archivados_atom_2.8.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )