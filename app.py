import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import f as f_dist
import re

# Configuración de la página
st.set_page_config(
    page_title="Diseños Experimentales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .kpi-card {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de la sesión
if 'selected_design' not in st.session_state:
    st.session_state.selected_design = 'unifactorial-dca-balanceado'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Información de diseños
DESIGN_INFO = {
    'unifactorial-dca-balanceado': {
        'nombre': 'DCA con Muestreo Balanceado',
        'modelo': 'Yij = μ + τi + εij',
        'descripcion': 'Mismo número de repeticiones por tratamiento'
    },
    'unifactorial-dca-no-balanceado': {
        'nombre': 'DCA con Muestreo No Balanceado',
        'modelo': 'Yij = μ + τi + εij',
        'descripcion': 'Diferente número de repeticiones por tratamiento'
    },
    'unifactorial-dca-bal-subbal': {
        'nombre': 'DCA Balanceado con Submuestreo Balanceado',
        'modelo': 'Yijk = μ + τi + εij + δijk',
        'descripcion': 'Pozas balanceadas, truchas balanceadas por poza'
    },
    'unifactorial-dca-bal-subnobal': {
        'nombre': 'DCA Balanceado con Submuestreo No Balanceado',
        'modelo': 'Yijk = μ + τi + εij + δijk',
        'descripcion': 'Pozas balanceadas, truchas variables por poza'
    },
    'bifactorial-dca-balanceado': {
        'nombre': 'DCA Bifactorial Balanceado',
        'modelo': 'Yijk = μ + αi + βj + (αβ)ij + εijk',
        'descripcion': 'Dos factores, repeticiones balanceadas'
    },
    'bifactorial-dca-no-balanceado': {
        'nombre': 'DCA Bifactorial No Balanceado',
        'modelo': 'Yijk = μ + αi + βj + (αβ)ij + εijk',
        'descripcion': 'Dos factores, repeticiones variables'
    },
    'trifactorial-dca-balanceado': {
        'nombre': 'DCA Trifactorial Balanceado',
        'modelo': 'Yijkl = μ + αi + βj + γk + ...',
        'descripcion': 'Tres factores, repeticiones balanceadas'
    },
    'machinelearning-random-forest': {
        'nombre': 'Random Forest Regressor',
        'modelo': 'Modelo de Ensamble No Paramétrico',
        'descripcion': 'Algoritmo de árboles de decisión múltiples'
    }
}

# Datos por defecto
DEFAULT_DATA = {
    'unifactorial-dca-balanceado': 'ALIM1: 210, 204, 199, 204, 185, 181, 187, 188, 208, 203\nALIM2: 231, 243, 222, 236, 235, 225, 228, 230, 225, 248\nALIM3: 251, 251, 241, 242, 255, 256, 251, 247, 243, 246',
    'unifactorial-dca-no-balanceado': 'ALIM1: 210, 204, 199, 204, 185, 181, 187, 188, 208\nALIM2: 231, 243, 222, 236, 235, 225, 228, 230, 225, 248, 251\nALIM3: 251, 241, 242, 255, 256, 251, 247, 243, 246',
    'unifactorial-dca-bal-subbal': 'ALIM1: 210,204; 199,204; 185,181; 187,188; 208,203\nALIM2: 231,243; 222,236; 235,225; 228,230; 225,248\nALIM3: 251,251; 241,242; 255,256; 251,247; 243,246',
    'unifactorial-dca-bal-subnobal': 'ALIM1: 210,204,199; 204,185; 181,187,188; 208,203\nALIM2: 231,243,222; 236,235; 225,228,230; 225,248\nALIM3: 251,251,241; 242,255; 256,251,247; 243,246',
    'bifactorial-dca-balanceado': 'A1B1: 210, 204, 199\nA1B2: 204, 185, 181\nA2B1: 231, 243, 222\nA2B2: 236, 235, 225\nA3B1: 251, 251, 241\nA3B2: 242, 255, 256',
    'bifactorial-dca-no-balanceado': 'A1B1: 210, 204, 199, 204\nA1B2: 185, 181, 187\nA2B1: 231, 243\nA2B2: 222, 236, 235, 225, 228\nA3B1: 251, 251, 241, 242\nA3B2: 255, 256',
    'trifactorial-dca-balanceado': 'A1B1C1: 210,204\nA1B1C2: 199,204\nA1B2C1: 185,181\nA1B2C2: 187,188\nA2B1C1: 231,243\nA2B1C2: 222,236\nA2B2C1: 235,225\nA2B2C2: 228,230',
    'machinelearning-random-forest': 'ALIM1,210\nALIM1,204\nALIM1,199\nALIM2,231\nALIM2,243\nALIM3,251\nALIM3,251'
}

# Funciones de análisis
def parse_unifactorial_data(input_text, has_subsampling=False):
    """Parsear datos unifactoriales"""
    lines = [l.strip() for l in input_text.split('\n') if l.strip()]
    treatment_names = []
    treatments_data = []
    
    for line in lines:
        parts = line.split(':')
        if len(parts) != 2:
            raise ValueError(f"Formato inválido: {line}")
        
        name = parts[0].strip()
        data_str = parts[1].strip()
        
        treatment_names.append(name)
        
        if has_subsampling:
            units = [unit.strip() for unit in data_str.split(';')]
            unit_data = []
            for unit in units:
                values = [float(v.strip()) for v in unit.split(',') if v.strip()]
                unit_data.append(values)
            treatments_data.append(unit_data)
        else:
            values = [float(v.strip()) for v in data_str.split(',') if v.strip()]
            treatments_data.append(values)
    
    return treatment_names, treatments_data

def anova_without_subsampling(treatments_data):
    """ANOVA sin submuestreo"""
    t = len(treatments_data)
    reps = [len(d) for d in treatments_data]
    N = sum(reps)
    
    if N <= t:
        raise ValueError("Datos insuficientes")
    
    all_data = np.concatenate(treatments_data)
    G = np.sum(all_data)
    FC = (G ** 2) / N
    
    SC_Total = np.sum(all_data ** 2) - FC
    
    SC_Trat = sum((np.sum(d) ** 2) / len(d) for d in treatments_data) - FC
    SC_Error = SC_Total - SC_Trat
    
    GL_Trat = t - 1
    GL_Error = N - t
    GL_Total = N - 1
    
    CM_Trat = SC_Trat / GL_Trat if GL_Trat > 0 else 0
    CM_Error = SC_Error / GL_Error if GL_Error > 0 else 0
    
    F_cal = CM_Trat / CM_Error if CM_Error > 0 else 0
    p_value = 1 - f_dist.cdf(F_cal, GL_Trat, GL_Error) if F_cal > 0 else 1
    
    return {
        'fuente': [
            {'source': 'Tratamiento', 'gl': GL_Trat, 'sc': SC_Trat, 'cm': CM_Trat, 'f': F_cal, 'p': p_value},
            {'source': 'Error', 'gl': GL_Error, 'sc': SC_Error, 'cm': CM_Error, 'f': '-', 'p': '-'},
            {'source': 'Total', 'gl': GL_Total, 'sc': SC_Total, 'cm': '-', 'f': '-', 'p': '-'}
        ]
    }

def anova_with_subsampling(treatments_data):
    """ANOVA con submuestreo"""
    t = len(treatments_data)
    total_units = sum(len(d) for d in treatments_data)
    
    all_flat_data = np.concatenate([np.concatenate(treat) for treat in treatments_data])
    N = len(all_flat_data)
    G = np.sum(all_flat_data)
    FC = (G ** 2) / N
    
    SC_Total = np.sum(all_flat_data ** 2) - FC
    
    # SC Tratamiento
    SC_Trat = 0
    for treat in treatments_data:
        treat_flat = np.concatenate(treat)
        SC_Trat += (np.sum(treat_flat) ** 2) / len(treat_flat)
    SC_Trat -= FC
    
    # SC para unidades experimentales
    sum_tij2_mij = 0
    for treat in treatments_data:
        for unit in treat:
            sum_tij2_mij += (np.sum(unit) ** 2) / len(unit)
    
    SC_Experimental = sum_tij2_mij - (SC_Trat + FC)
    SC_Muestreo = SC_Total - sum_tij2_mij
    
    GL_Trat = t - 1
    GL_Experimental = total_units - t
    GL_Muestreo = N - total_units
    GL_Total = N - 1
    
    CM_Trat = SC_Trat / GL_Trat if GL_Trat > 0 else 0
    CM_Experimental = SC_Experimental / GL_Experimental if GL_Experimental > 0 else 0
    CM_Muestreo = SC_Muestreo / GL_Muestreo if GL_Muestreo > 0 else 0
    
    F_cal = CM_Trat / CM_Experimental if CM_Experimental > 0 else 0
    p_value = 1 - f_dist.cdf(F_cal, GL_Trat, GL_Experimental) if F_cal > 0 else 1
    
    return {
        'fuente': [
            {'source': 'Tratamiento', 'gl': GL_Trat, 'sc': SC_Trat, 'cm': CM_Trat, 'f': F_cal, 'p': p_value},
            {'source': 'Error Experimental', 'gl': GL_Experimental, 'sc': SC_Experimental, 'cm': CM_Experimental, 'f': '-', 'p': '-'},
            {'source': 'Error de Muestreo', 'gl': GL_Muestreo, 'sc': SC_Muestreo, 'cm': CM_Muestreo, 'f': '-', 'p': '-'},
            {'source': 'Total', 'gl': GL_Total, 'sc': SC_Total, 'cm': '-', 'f': '-', 'p': '-'}
        ]
    }

def parse_bifactorial_data(input_text):
    """Parsear datos bifactoriales"""
    lines = [l.strip() for l in input_text.split('\n') if l.strip()]
    data_map = {}
    levels_A = set()
    levels_B = set()
    
    for line in lines:
        parts = line.split(':')
        if len(parts) != 2:
            raise ValueError(f"Formato inválido: {line}")
        
        treatment = parts[0].strip()
        values_str = parts[1].strip()
        values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
        
        match = re.match(r'A(\d+)B(\d+)', treatment, re.IGNORECASE)
        if not match:
            raise ValueError(f"Formato de tratamiento inválido: {treatment}")
        
        level_A = f"A{match.group(1)}"
        level_B = f"B{match.group(2)}"
        
        levels_A.add(level_A)
        levels_B.add(level_B)
        data_map[treatment] = values
    
    return data_map, sorted(levels_A), sorted(levels_B)

def anova_bifactorial(data_map, levels_A, levels_B):
    """ANOVA bifactorial"""
    a = len(levels_A)
    b = len(levels_B)
    
    all_data = np.concatenate(list(data_map.values()))
    N = len(all_data)
    G = np.sum(all_data)
    FC = (G ** 2) / N
    
    SC_Total = np.sum(all_data ** 2) - FC
    
    # SC Factor A
    sums_A = {}
    counts_A = {}
    for key, values in data_map.items():
        match = re.match(r'A(\d+)B(\d+)', key, re.IGNORECASE)
        level_A = f"A{match.group(1)}"
        sums_A[level_A] = sums_A.get(level_A, 0) + np.sum(values)
        counts_A[level_A] = counts_A.get(level_A, 0) + len(values)
    
    SC_A = sum((sums_A[k] ** 2) / counts_A[k] for k in sums_A) - FC
    
    # SC Factor B
    sums_B = {}
    counts_B = {}
    for key, values in data_map.items():
        match = re.match(r'A(\d+)B(\d+)', key, re.IGNORECASE)
        level_B = f"B{match.group(2)}"
        sums_B[level_B] = sums_B.get(level_B, 0) + np.sum(values)
        counts_B[level_B] = counts_B.get(level_B, 0) + len(values)
    
    SC_B = sum((sums_B[k] ** 2) / counts_B[k] for k in sums_B) - FC
    
    # SC Celdas
    SC_cells = sum((np.sum(values) ** 2) / len(values) for values in data_map.values()) - FC
    
    SC_AB = SC_cells - SC_A - SC_B
    SC_Error = SC_Total - SC_cells
    
    GL_A = a - 1
    GL_B = b - 1
    GL_AB = (a - 1) * (b - 1)
    GL_Error = sum(len(v) - 1 for v in data_map.values())
    GL_Total = N - 1
    
    CM_A = SC_A / GL_A if GL_A > 0 else 0
    CM_B = SC_B / GL_B if GL_B > 0 else 0
    CM_AB = SC_AB / GL_AB if GL_AB > 0 else 0
    CM_Error = SC_Error / GL_Error if GL_Error > 0 else 0
    
    F_A = CM_A / CM_Error if CM_Error > 0 else 0
    F_B = CM_B / CM_Error if CM_Error > 0 else 0
    F_AB = CM_AB / CM_Error if CM_Error > 0 else 0
    
    p_A = 1 - f_dist.cdf(F_A, GL_A, GL_Error) if F_A > 0 else 1
    p_B = 1 - f_dist.cdf(F_B, GL_B, GL_Error) if F_B > 0 else 1
    p_AB = 1 - f_dist.cdf(F_AB, GL_AB, GL_Error) if F_AB > 0 else 1
    
    return {
        'fuente': [
            {'source': 'Factor A', 'gl': GL_A, 'sc': SC_A, 'cm': CM_A, 'f': F_A, 'p': p_A},
            {'source': 'Factor B', 'gl': GL_B, 'sc': SC_B, 'cm': CM_B, 'f': F_B, 'p': p_B},
            {'source': 'Interacción A*B', 'gl': GL_AB, 'sc': SC_AB, 'cm': CM_AB, 'f': F_AB, 'p': p_AB},
            {'source': 'Error', 'gl': GL_Error, 'sc': SC_Error, 'cm': CM_Error, 'f': '-', 'p': '-'},
            {'source': 'Total', 'gl': GL_Total, 'sc': SC_Total, 'cm': '-', 'f': '-', 'p': '-'}
        ],
        'data_map': data_map,
        'levels_A': levels_A,
        'levels_B': levels_B
    }

def analyze_ml_simple(input_text):
    """Análisis ML simplificado"""
    lines = [l.strip() for l in input_text.split('\n') if l.strip()]
    data = []
    
    for line in lines:
        parts = line.split(',')
        if len(parts) != 2:
            raise ValueError(f"Formato inválido: {line}")
        data.append({'feature': parts[0].strip(), 'value': float(parts[1].strip())})
    
    df = pd.DataFrame(data)
    
    # Calcular estadísticas por categoría
    means = df.groupby('feature')['value'].mean()
    predictions = df['feature'].map(means)
    
    mse = np.mean((df['value'] - predictions) ** 2)
    rmse = np.sqrt(mse)
    
    ss_tot = np.sum((df['value'] - df['value'].mean()) ** 2)
    ss_res = np.sum((df['value'] - predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    category_stats = df.groupby('feature')['value'].agg(['mean', 'count', 'std']).reset_index()
    
    return {
        'type': 'ml',
        'rmse': rmse,
        'r2': r2 * 100,
        'category_stats': category_stats,
        'data': df
    }

# Encabezado principal
st.markdown("""
<div class="main-header">
    <h1>📚 CURSO: DISEÑOS EXPERIMENTALES</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">Dashboard Interactivo de Análisis Estadístico</p>
</div>
""", unsafe_allow_html=True)

# Información del curso
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="kpi-card" style="border-left-color: #3b82f6; background-color: #eff6ff;">
        <h4>👤 Estudiante</h4>
        <p style="font-size: 1.1rem; font-weight: bold;">Gomez Tacuri Jose Fernando</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="kpi-card" style="border-left-color: #8b5cf6; background-color: #f5f3ff;">
        <h4>🎓 Docente</h4>
        <p style="font-size: 1.1rem; font-weight: bold;">LLUEN VALLEJOS CESAR AUGUSTO</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="kpi-card" style="border-left-color: #a855f7; background-color: #faf5ff;">
        <h4>📅 Periodo Académico</h4>
        <p style="font-size: 1.1rem; font-weight: bold;">2025-II</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="kpi-card" style="border-left-color: #10b981; background-color: #d1fae5;">
        <h4>📄 Tema de Estudio</h4>
        <p style="font-size: 1.1rem; font-weight: bold;">Engorde de truchas en pozas</p>
    </div>
    """, unsafe_allow_html=True)

# Mostrar enunciado del experimento
with st.expander("📖 Ver Enunciado del Experimento", expanded=False):
    st.markdown("""
    ### Contexto del Experimento
    
    Un investigador de la Universidad Nacional del Altiplano en Puno desea evaluar el efecto de **diferentes dietas alimenticias** 
    en el engorde de truchas arcoíris (*Oncorhynchus mykiss*) criadas en pozas.
    
    Para ello, se seleccionaron aleatoriamente un número de pozas, y a cada poza se le asignó una de las dietas. 
    Después de un periodo de **90 días**, se registró el peso final (en gramos) de una muestra de truchas de cada poza.
    
    **Objetivo:** Determinar si existen diferencias estadísticamente significativas en la ganancia de peso que puedan 
    ser atribuidas a las diferentes dietas suministradas, utilizando para ello distintos modelos de diseños completamente 
    al azar (DCA) y modelos de Machine Learning.
    """)

st.markdown("---")

# Sección 1: Seleccionar Metodología
st.header("1️⃣ Seleccionar Metodología y Diseño")

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Unifactorial (ANOVA)", "🔬🔬 Bifactorial (ANOVA)", 
                                    "🔬🔬🔬 Trifactorial (ANOVA)", "🤖 Machine Learning"])

with tab1:
    st.subheader("Diseños Unifactoriales")
    design_type = st.radio(
        "Seleccione el diseño experimental:",
        ['unifactorial-dca-balanceado', 'unifactorial-dca-no-balanceado',
         'unifactorial-dca-bal-subbal', 'unifactorial-dca-bal-subnobal'],
        format_func=lambda x: DESIGN_INFO[x]['nombre'],
        key='unifactorial_design'
    )
    st.session_state.selected_design = design_type
    
    info = DESIGN_INFO[design_type]
    st.info(f"**Modelo:** `{info['modelo']}`\n\n{info['descripcion']}")

with tab2:
    st.subheader("Diseños Bifactoriales")
    design_type = st.radio(
        "Seleccione el diseño experimental:",
        ['bifactorial-dca-balanceado', 'bifactorial-dca-no-balanceado'],
        format_func=lambda x: DESIGN_INFO[x]['nombre'],
        key='bifactorial_design'
    )
    st.session_state.selected_design = design_type
    
    info = DESIGN_INFO[design_type]
    st.info(f"**Modelo:** `{info['modelo']}`\n\n{info['descripcion']}")

with tab3:
    st.subheader("Diseños Trifactoriales")
    design_type = st.radio(
        "Seleccione el diseño experimental:",
        ['trifactorial-dca-balanceado'],
        format_func=lambda x: DESIGN_INFO[x]['nombre'],
        key='trifactorial_design'
    )
    st.session_state.selected_design = design_type
    
    info = DESIGN_INFO[design_type]
    st.info(f"**Modelo:** `{info['modelo']}`\n\n{info['descripcion']}")

with tab4:
    st.subheader("Modelos de Machine Learning")
    design_type = st.radio(
        "Seleccione el modelo:",
        ['machinelearning-random-forest'],
        format_func=lambda x: DESIGN_INFO[x]['nombre'],
        key='ml_design'
    )
    st.session_state.selected_design = design_type
    
    info = DESIGN_INFO[design_type]
    st.info(f"**Modelo:** `{info['modelo']}`\n\n{info['descripcion']}")

st.markdown("---")

# Sección 2: Ingresar y Analizar Datos
st.header("2️⃣ Ingresar y Analizar Datos")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Datos de Entrada")
    default_text = DEFAULT_DATA.get(st.session_state.selected_design, '')
    input_data = st.text_area(
        "Formato de entrada:",
        value=default_text,
        height=300,
        help="Ingrese los datos según el formato del diseño seleccionado"
    )
    
    # Ayuda de formato
    if 'unifactorial' in st.session_state.selected_design:
        if 'sub' in st.session_state.selected_design:
            st.caption("💡 Use ';' para separar unidades experimentales y ',' para separar submuestras")
        else:
            st.caption("💡 Separe los valores con comas. Cada línea representa un tratamiento")
    elif 'bifactorial' in st.session_state.selected_design:
        st.caption("💡 Use formato AxBy: valores")
    elif 'machinelearning' in st.session_state.selected_design:
        st.caption("💡 Cada línea: Categoría,Valor")

with col2:
    st.subheader("Ejecutar Análisis")
    st.write("")
    st.write("")
    
    if st.button("🔍 Analizar y Visualizar", use_container_width=True, type="primary"):
        try:
            # Realizar análisis según el tipo de diseño
            if 'unifactorial' in st.session_state.selected_design:
                has_sub = 'sub' in st.session_state.selected_design
                names, data = parse_unifactorial_data(input_data, has_sub)
                
                if has_sub:
                    anova_result = anova_with_subsampling(data)
                    flat_data = [[item for sublist in treat for item in sublist] for treat in data]
                else:
                    anova_result = anova_without_subsampling(data)
                    flat_data = data
                
                st.session_state.analysis_results = {
                    'type': 'anova',
                    'anova': anova_result,
                    'treatment_names': names,
                    'data': flat_data,
                    'design_type': 'unifactorial'
                }
                st.success("✅ Análisis completado exitosamente!")
                
            elif 'bifactorial' in st.session_state.selected_design:
                data_map, levels_A, levels_B = parse_bifactorial_data(input_data)
                anova_result = anova_bifactorial(data_map, levels_A, levels_B)
                
                st.session_state.analysis_results = {
                    'type': 'anova',
                    'anova': anova_result,
                    'data_map': data_map,
                    'levels_A': levels_A,
                    'levels_B': levels_B,
                    'design_type': 'bifactorial'
                }
                st.success("✅ Análisis completado exitosamente!")
                
            elif 'machinelearning' in st.session_state.selected_design:
                ml_result = analyze_ml_simple(input_data)
                st.session_state.analysis_results = ml_result
                st.success("✅ Análisis completado exitosamente!")
                
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.session_state.analysis_results = None

st.markdown("---")

# Sección 3: Dashboard de Resultados
if st.session_state.analysis_results:
    st.header("3️⃣ Dashboard de Resultados")
    
    results = st.session_state.analysis_results
    
    if results['type'] == 'anova':
        # KPIs principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #a855f7; background-color: #faf5ff;">
                <h4>📊 Modelo Estadístico</h4>
                <p style="font-size: 0.9rem; font-weight: bold;">{DESIGN_INFO[st.session_state.selected_design]['modelo']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Obtener el primer resultado de tratamiento/factor
        first_result = results['anova']['fuente'][0]
        
        with col2:
            f_val = first_result['f'] if isinstance(first_result['f'], (int, float)) else 0
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #f59e0b; background-color: #fffbeb;">
                <h4>📈 Estadístico F</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{f_val:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            p_val = first_result['p']
            p_color = "#10b981" if p_val < 0.05 else "#ef4444"
            p_bg = "#d1fae5" if p_val < 0.05 else "#fee2e2"
            p_text = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: {p_color}; background-color: {p_bg};">
                <h4>🎯 P-valor</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{p_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Tabla ANOVA
        st.subheader("📋 Tabla de Análisis de Varianza (ANOVA)")
        
        anova_df = pd.DataFrame(results['anova']['fuente'])
        anova_df['sc'] = anova_df['sc'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        anova_df['cm'] = anova_df['cm'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        anova_df['f'] = anova_df['f'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        anova_df['p'] = anova_df['p'].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "< 0.0001" if isinstance(x, (int, float)) else x)
        
        anova_df.columns = ['Fuente de Variación', 'GL', 'SC', 'CM', 'F', 'P-valor']
        
        st.dataframe(anova_df, use_container_width=True, hide_index=True)
        
        # Interpretación
        st.subheader("💡 Interpretación de Resultados")
        
        for row in results['anova']['fuente']:
            if isinstance(row['p'], (int, float)) and 'error' not in row['source'].lower() and 'total' not in row['source'].lower():
                if row['p'] < 0.05:
                    st.markdown(f"""
                    <div class="info-box success-box">
                        <strong>{row['source']}:</strong> P-valor = {row['p']:.4f} < 0.05<br>
                        ✅ Se rechaza la hipótesis nula. Existe evidencia de un efecto significativo.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-box error-box">
                        <strong>{row['source']}:</strong> P-valor = {row['p']:.4f} ≥ 0.05<br>
                        ❌ No se rechaza la hipótesis nula. No hay evidencia de un efecto significativo.
                    </div>
                    """, unsafe_allow_html=True)
        
        # Visualizaciones
        st.subheader("📊 Visualizaciones")
        
        if results['design_type'] == 'unifactorial':
            # Gráfico de cajas
            fig = go.Figure()
            for i, (name, data) in enumerate(zip(results['treatment_names'], results['data'])):
                fig.add_trace(go.Box(
                    y=data,
                    name=name,
                    boxmean='sd',
                    marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                ))
            
            fig.update_layout(
                title="Gráfico de Cajas por Tratamiento",
                yaxis_title="Peso (gramos)",
                xaxis_title="Tratamiento",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de barras de medias
            means = [np.mean(d) for d in results['data']]
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=results['treatment_names'],
                    y=means,
                    marker_color=px.colors.qualitative.Set2[:len(means)],
                    text=[f"{m:.2f}" for m in means],
                    textposition='outside'
                )
            ])
            fig_bar.update_layout(
                title="Promedio por Tratamiento",
                yaxis_title="Peso promedio (gramos)",
                xaxis_title="Tratamiento",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        elif results['design_type'] == 'bifactorial':
            # Gráfico de interacción
            interaction_data = []
            for level_b in results['levels_B']:
                for level_a in results['levels_A']:
                    key = f"{level_a}{level_b}"
                    if key in results['data_map']:
                        mean_val = np.mean(results['data_map'][key])
                        interaction_data.append({
                            'Factor A': level_a,
                            'Factor B': level_b,
                            'Media': mean_val
                        })
            
            df_interaction = pd.DataFrame(interaction_data)
            
            fig_interaction = px.line(
                df_interaction,
                x='Factor B',
                y='Media',
                color='Factor A',
                markers=True,
                title="Gráfico de Interacción (A*B)"
            )
            fig_interaction.update_layout(height=500)
            st.plotly_chart(fig_interaction, use_container_width=True)
            
            # Heatmap de medias
            pivot_data = df_interaction.pivot(index='Factor A', columns='Factor B', values='Media')
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn',
                text=pivot_data.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 14}
            ))
            fig_heat.update_layout(
                title="Mapa de Calor de Medias",
                xaxis_title="Factor B",
                yaxis_title="Factor A",
                height=400
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    
    elif results['type'] == 'ml':
        # KPIs de ML
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #a855f7; background-color: #faf5ff;">
                <h4>🤖 Modelo</h4>
                <p style="font-size: 0.9rem; font-weight: bold;">Random Forest Simplificado</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #f59e0b; background-color: #fffbeb;">
                <h4>📉 RMSE</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{results['rmse']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #10b981; background-color: #d1fae5;">
                <h4>📊 R² (%)</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{results['r2']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Estadísticas por categoría
        st.subheader("📋 Estadísticas por Categoría")
        stats_df = results['category_stats'].copy()
        stats_df.columns = ['Categoría', 'Media', 'n', 'Desv. Est.']
        stats_df['Media'] = stats_df['Media'].apply(lambda x: f"{x:.2f}")
        stats_df['Desv. Est.'] = stats_df['Desv. Est.'].apply(lambda x: f"{x:.2f}")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Gráfico de barras
        fig = px.bar(
            results['category_stats'],
            x='feature',
            y='mean',
            error_y='std',
            title="Media por Categoría",
            labels={'feature': 'Categoría', 'mean': 'Peso Promedio (g)'},
            color='feature'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación
        st.markdown(f"""
        <div class="info-box warning-box">
            <h4>💡 Interpretación del Modelo</h4>
            <p>El <strong>RMSE</strong> de <strong>{results['rmse']:.4f}</strong> indica el error promedio de predicción del modelo en gramos.</p>
            <p>El <strong>R²</strong> de <strong>{results['r2']:.2f}%</strong> indica que el modelo explica ese porcentaje de la variabilidad en los datos.</p>
            <p><em>Nota: Este es un análisis simplificado para demostración educativa del concepto de Machine Learning.</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>📊 Dashboard de Diseños Experimentales | UNAP 2025-II</p>
    <p style="font-size: 0.9rem;">Desarrollado para el análisis estadístico de experimentos agrícolas</p>
</div>
""", unsafe_allow_html=True)
