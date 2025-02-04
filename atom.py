# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:45:36 2025

@author: jveraz
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from openai import OpenAI

# Configurar la aplicación Streamlit
st.title("Archivador Inteligente de Documentos Antiguos")

# Inicializar cliente de OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def generate_metadata(description, field):
    """Utiliza OpenAI para generar metadatos específicos según el campo del formato AtoM 2.8."""
    prompt = (
        f"Eres un archivista experto en catalogación de documentos históricos usando el formato AtoM 2.8. "
        f"Genera contenido para el campo '{field}' con base en la siguiente descripción del documento: {description}"
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
    for column in df.columns:
        for atom_field in atom_template.columns:
            if column.lower().replace(" ", "_") in atom_field.lower().replace(" ", "_"):
                output_df[atom_field] = df[column]
                break
    
    # Generar metadatos adicionales según los campos del formato AtoM 2.8
    st.write("Generando metadatos específicos del formato AtoM 2.8...")
    for field in output_df.columns:
        if output_df[field].isnull().all():
            continue
        output_df[field] = output_df[field].apply(lambda x: generate_metadata(str(x), field) if pd.notna(x) else "")
    
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
