import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
# Nota: La funcionalidad del stemmer depende de que NLTK esté disponible en el entorno.
from nltk.stem import SnowballStemmer 

# --- CSS GÓTICO VARIACIÓN: ÉNFASIS EN EL CONTRASTE DE DATOS ---
gothic_css_variant = """
<style>
/* Paleta base: Fondo #111111, Texto #E0E0E0 (Pergamino ligero), Acento #5A4832 (Bronce/Metal), Sangre #A50000 */
.stApp {
    background-color: #111111;
    color: #E0E0E0;
    font-family: 'Georgia', serif;
}

/* Título Principal (h1) */
h1 {
    color: #A50000; /* Rojo sangre */
    text-shadow: 3px 3px 8px #000000;
    font-size: 3.2em; 
    border-bottom: 5px solid #5A4832; /* Borde Bronce */
    padding-bottom: 10px;
    margin-bottom: 30px;
    text-align: center;
    letter-spacing: 2px;
}

/* Subtítulos (h2, h3): Énfasis en el bronce */
h2, h3 {
    color: #C0C0C0; /* Plata/gris claro */
    border-left: 5px solid #5A4832;
    padding-left: 10px;
    margin-top: 25px;
}

/* Input y Text Area (El Papiro de Inscripción) */
div[data-testid="stTextInput"], div[data-testid="stTextarea"] {
    background-color: #1A1A1A;
    border: 1px solid #5A4832;
    padding: 10px;
    border-radius: 5px;
    color: #F5F5DC;
}

/* Botones (Forjar, Iniciar): Botones tipo Sello/Relicario */
.stButton > button {
    background-color: #303030; /* Base oscura */
    color: #E0E0E0;
    border: 2px solid #A50000; /* Borde rojo sangre */
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 4px 4px 8px #000000, inset 0 0 5px rgba(255, 255, 255, 0.05);
    transition: background-color 0.3s, transform 0.1s;
}

.stButton > button:hover {
    background-color: #A50000; /* Hover a rojo intenso */
    color: white;
    transform: translateY(-1px);
}

/* Dataframe (El Mapa Estelar de las Runas) */
div[data-testid="stDataFrame"] table {
    background-color: #1A1A1A;
    border: 1px solid #5A4832;
    color: #E0E0E0;
}
div[data-testid="stDataFrame"] thead tr th {
    background-color: #2A2A2A !important;
    color: #A50000 !important;
}

/* Texto de Alertas (Revelaciones) */
.stSuccess { background-color: #20251B; color: #F5F5DC; border-left: 5px solid #5A4832; }
.stInfo { background-color: #1A1A25; color: #F5F5DC; border-left: 5px solid #5A4832; }
.stWarning { background-color: #352A1A; color: #F5F5DC; border-left: 5px solid #A50000; }
</style>
"""
st.markdown(gothic_css_variant, unsafe_allow_html=True)

# Título GÓTICO
st.title("El Códice de Resonancia Profana (TF-IDF)")

st.write("""
**Los Secretos del Análisis de la Afinidad:**

Cada línea se consagra como un **Fragmento del Códice**. La Consulta debe estar inscrita en la lengua arcana del **Inglés**, ya que el Oráculo se ha sintonizado para ese idioma.

Este artefacto aplica la **Normalización y la Reducción a la Raíz (Stemming)**, asegurando que las distintas formas de una runa (ej. *playing* y *play*) sean consideradas como equivalentes en su potencia mística.
""")

# Ejemplo inicial en inglés (adaptado a la narrativa gótica)
default_docs = "The watchdog howls under the chilling moon.\nThe feral cat observes the spectral transit of the night.\nThe hound and the feline perform the rite together."

text_input = st.text_area(
    "📜 Los Fragmentos del Códice (uno por línea, en Inglés):",
    default_docs,
    height=150
)

# Inicializar estado de sesión para la pregunta
if 'english_question' not in st.session_state:
    st.session_state.english_question = "Which creature howls?"

st.session_state.english_question = st.text_input(
    "🕯️ El Susurro de la Consulta (Pregunta en Inglés):", 
    st.session_state.english_question
)

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    """El Decodificador de Runas: Limpia, tokeniza y reduce las palabras a su raíz (stem)."""
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos (manteniendo solo a-z)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🔮 Desentrañar la Afinidad (Calcular Resonancia)"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    question = st.session_state.english_question
    
    if len(documents) < 1:
        st.error("⚠️ El Códice requiere al menos un Fragmento para el Ritual.")
    elif not question.strip():
        st.error("⚠️ El Susurro de la Consulta no puede ser un vacío espectral.")
    else:
        # Vectorizador con stemming y stop words en inglés
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None # Necesario cuando se usa un tokenizer personalizado
        )

        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)

        # Mostrar matriz TF-IDF
        st.markdown("### 🗺️ El Mapa Estelar de las Runas (Matriz TF-IDF)")
        st.caption("Cada valor indica el peso místico de una Runa Base (*Stem*) dentro de un Fragmento.")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Fragmento {i+1}" for i in range(len(documents))]
        )

        st.dataframe(df_tfidf.round(4), use_container_width=True)

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno (Afinidad)
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 🎯 La Revelación Afín (El Fragmento más Cercano)")
        st.markdown(f"**El Susurro que Busca:** *{question}*")
        
        # Uso de iconos en las alertas
        if best_score > 0.3:
            st.success(f"**Respuesta Revelada (Fragmento {best_idx+1}):** {best_doc}")
            st.info(f"**Nivel de Afinidad Mística:** {best_score:.4f}")
        else:
            st.warning(f"**Respuesta (Baja Resonancia):** {best_doc}")
            st.info(f"**Nivel de Afinidad Mística:** {best_score:.4f} (Se requiere un Susurro más preciso)")

        st.markdown("### 📊 Tabulación de Resonancias por Fragmento")
        sim_df = pd.DataFrame({
            "Fragmento": [f"Fragmento {i+1}" for i in range(len(documents))],
            "Texto del Códice": documents,
            "Nivel de Afinidad": similarities.round(4)
        })
        st.dataframe(sim_df.sort_values("Nivel de Afinidad", ascending=False), use_container_width=True)

        st.markdown("### 🗝️ Sellos de la Consulta Presentes en el Fragmento Elegido")
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        # Mostrar los stems que existen en el vocabulario general Y tienen peso (>0) en el documento con mayor afinidad
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        
        if matched:
            st.markdown(f"**Runas Raíz (Stems) encontradas:** `{', '.join(matched)}`")
        else:
            st.warning("Ninguna Runa Raíz de la Consulta fue hallada en el Fragmento elegido.")




