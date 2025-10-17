import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import f as f_dist
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Diseños Experimentales",
    page_icon="📊",
    layout="wide"
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
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar datos de ejemplo
DEFAULT_DATA = {
    'unifactorial-dca-balanceado': 'ALIM1: 210, 204, 199, 204, 185, 181, 187, 188, 208, 203\nALIM2: 231, 243, 222, 236, 235, 225, 228, 230, 225, 248\nALIM3: 251, 251, 241, 242, 255, 256, 251, 247, 243, 246',
    'unifactorial-dca-no-balanceado': 'ALIM1: 210, 204, 199, 204, 185, 181, 187, 188, 208\nALIM2: 231, 243, 222, 236, 235, 225, 228, 230, 225, 248, 251\nALIM3: 251, 241, 242, 255, 256, 251, 247, 243, 246',
    'bifactorial-dca-balanceado': 'A1B1: 210, 204, 199\nA1B2: 204, 185, 181\nA2B1: 231, 243, 222\nA2B2: 236, 235, 225\nA3B1: 251, 251, 241\nA3B2: 242, 255, 256',
    'bifactorial-dca-no-balanceado': 'A1B1: 210, 204, 199, 204\nA1B2: 185, 181, 187\nA2B1: 231, 243\nA2B2: 222, 236, 235, 225, 228\nA3B1: 251, 251, 241, 242\nA3B2: 255, 256',
}

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
}

def parse_unifactorial_data(input_text):
    """Parsear datos unifactoriales"""
    lines = [l.strip() for l in input_text.split('\n') if l.strip()]
    treatment_names = []
    treatments_data = []
    
    for line in lines:
        parts = line.split(':')
        if len(parts) != 2:
            raise ValueError(f"Formato inválido en línea: {line}")
        
        name = parts[0].strip()
        values_str = parts[1].strip()
        values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
        
        treatment_names.append(name)
        treatments_data.append(values)
    
    return treatment_names, treatments_data

def parse_bifactorial_data(input_text):
    """Parsear datos bifactoriales"""
    lines = [l.strip() for l in input_text.split('\n') if l.strip()]
    data_map = {}
    levels_a = set()
    levels_b = set()
    
    for line in lines:
        parts = line.split(':')
        if len(parts) != 2:
            raise ValueError(f"Formato inválido en línea: {line}")
        
        treatment = parts[0].strip()
        values = [float(v.strip()) for v in parts[1].split(',') if v.strip()]
        
        # Extraer niveles A y B
        import re
        match = re.match(r'A(\d+)B(\d+)', treatment, re.IGNORECASE)
        if not match:
            raise ValueError(f"Formato de tratamiento inválido: {treatment}")
        
        level_a = f"A{match.group(1)}"
        level_b = f"B{match.group(2)}"
        
        levels_a.add(level_a)
        levels_b.add(level_b)
        data_map[treatment] = values
    
    return data_map, sorted(levels_a), sorted(levels_b)

def anova_unifactorial(treatments_data):
    """Realizar ANOVA unifactorial"""
    t = len(treatments_data)
    reps = [len(d) for d in treatments_data]
    N = sum(reps)
    
    if N <= t:
        raise ValueError("Datos insuficientes para el análisis")
    
    # Concatenar todos los datos
    all_data = [item for sublist in treatments_data for item in sublist]
    G = sum(all_data)
    FC = (G ** 2) / N
    
    # Suma de cuadrados total
    SC_Total = sum(y ** 2 for y in all_data) - FC
    
    # Suma de cuadrados tratamiento
    SC_Trat = 0
    for i, data in enumerate(treatments_data):
        if len(data) > 0:
            Ti = sum(data)
            SC_Trat += (Ti ** 2) / len(data)
    SC_Trat -= FC
    
    # Suma de cuadrados error
    SC_Error = SC_Total - SC_Trat
    
    # Grados de libertad
    GL_Trat = t - 1
    GL_Error = N - t
    GL_Total = N - 1
    
    # Cuadrados medios
    CM_Trat = SC_Trat / GL_Trat if GL_Trat > 0 else 0
    CM_Error = SC_Error / GL_Error if GL_Error > 0 else 0
    
    # Estadístico F y p-valor
    F_cal = CM_Trat / CM_Error if CM_Error > 0 else 0
    p_value = 1 - f_dist.cdf(F_cal, GL_Trat, GL_Error) if F_cal > 0 else 1
    
    return {
        'fuente': [
            {'source': 'Tratamiento', 'gl': GL_Trat, 'sc': SC_Trat, 'cm': CM_Trat, 'f': F_cal, 'p': p_value},
            {'source': 'Error', 'gl': GL_Error, 'sc': SC_Error, 'cm': CM_Error, 'f': '-', 'p': '-'},
            {'source': 'Total', 'gl': GL_Total, 'sc': SC_Total, 'cm': '-', 'f': '-', 'p': '-'}
        ]
    }

def anova_bifactorial(data_map, levels_a, levels_b):
    """Realizar ANOVA bifactorial"""
    a = len(levels_a)
    b = len(levels_b)
    
    # Todos los datos
    all_data = [item for values in data_map.values() for item in values]
    N = len(all_data)
    G = sum(all_data)
    FC = (G ** 2) / N
    
    # SC Total
    SC_Total = sum(y ** 2 for y in all_data) - FC
    
    # Calcular sumas por factor A
    sums_by_a = {}
    counts_by_a = {}
    for key, values in data_map.items():
        import re
        match = re.match(r'A(\d+)B(\d+)', key, re.IGNORECASE)
        level_a = f"A{match.group(1)}"
        sums_by_a[level_a] = sums_by_a.get(level_a, 0) + sum(values)
        counts_by_a[level_a] = counts_by_a.get(level_a, 0) + len(values)
    
    # SC Factor A
    SC_A = sum((s ** 2) / counts_by_a[k] for k, s in sums_by_a.items()) - FC
    
    # Calcular sumas por factor B
    sums_by_b = {}
    counts_by_b = {}
    for key, values in data_map.items():
        import re
        match = re.match(r'A(\d+)B(\d+)', key, re.IGNORECASE)
        level_b = f"B{match.group(2)}"
        sums_by_b[level_b] = sums_by_b.get(level_b, 0) + sum(values)
        counts_by_b[level_b] = counts_by_b.get(level_b, 0) + len(values)
    
    # SC Factor B
    SC_B = sum((s ** 2) / counts_by_b[k] for k, s in sums_by_b.items()) - FC
    
    # SC Celdas
    SC_cells = sum((sum(v) ** 2) / len(v) for v in data_map.values()) - FC
    
    # SC Interacción
    SC_AB = SC_cells - SC_A - SC_B
    
    # SC Error
    SC_Error = SC_Total - SC_cells
    
    # Grados de libertad
    GL_A = a - 1
    GL_B = b - 1
    GL_AB = (a - 1) * (b - 1)
    GL_Error = sum(len(v) - 1 for v in data_map.values())
    GL_Total = N - 1
    
    # Cuadrados medios
    CM_A = SC_A / GL_A if GL_A > 0 else 0
    CM_B = SC_B / GL_B if GL_B > 0 else 0
    CM_AB = SC_AB / GL_AB if GL_AB > 0 else 0
    CM_Error = SC_Error / GL_Error if GL_Error > 0 else 0
    
    # Estadísticos F
    F_A = CM_A / CM_Error if CM_Error > 0 else 0
    F_B = CM_B / CM_Error if CM_Error > 0 else 0
    F_AB = CM_AB / CM_Error if CM_Error > 0 else 0
    
    # P-valores
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
        'levels_a': levels_a,
        'levels_b': levels_b
    }

def create_boxplot(treatment_names, treatments_data):
    """Crear gráfico de cajas"""
    fig = go.Figure()
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#ef4444']
    
    for i, (name, data) in enumerate(zip(treatment_names, treatments_data)):
        fig.add_trace(go.Box(
            y=data,
            name=name,
            marker_color=colors[i % len(colors)],
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Gráfico de Cajas por Tratamiento",
        yaxis_title="Valor",
        xaxis_title="Tratamiento",
        height=400,
        showlegend=False
    )
    
    return fig

def create_interaction_plot(data_map, levels_a, levels_b):
    """Crear gráfico de interacción"""
    fig = go.Figure()
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']
    
    for i, level_a in enumerate(levels_a):
        means = []
        for level_b in levels_b:
            key = f"{level_a}{level_b}"
            if key in data_map:
                means.append(np.mean(data_map[key]))
            else:
                means.append(None)
        
        fig.add_trace(go.Scatter(
            x=levels_b,
            y=means,
            mode='lines+markers',
            name=level_a,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="Gráfico de Interacción (A*B)",
        xaxis_title="Factor B",
        yaxis_title="Media",
        height=400,
        legend_title="Factor A"
    )
    
    return fig

# ==================== INTERFAZ PRINCIPAL ====================

# Encabezado
st.markdown("""
<div class="main-header">
    <h1>📚 CURSO: DISEÑOS EXPERIMENTALES</h1>
    <p>Dashboard Interactivo para Análisis Estadístico</p>
</div>
""", unsafe_allow_html=True)

# Información del curso
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**👤 Estudiante**")
    st.info("Gomez Tacuri Jose Fernando")

with col2:
    st.markdown("**👨‍🏫 Docente**")
    st.info("LLUEN VALLEJOS CESAR AUGUSTO")

with col3:
    st.markdown("**📅 Periodo**")
    st.info("2025-II")

with col4:
    st.markdown("**📄 Tema**")
    if st.button("📋 Ver Enunciado"):
        with st.expander("**Enunciado del Experimento**", expanded=True):
            st.markdown("""
            Un investigador de la Universidad Nacional del Altiplano en Puno desea evaluar el efecto de 
            **diferentes dietas alimenticias** en el engorde de truchas arcoíris (*Oncorhynchus mykiss*) 
            criadas en pozas.
            
            Para ello, se seleccionaron aleatoriamente un número de pozas, y a cada poza se le asignó 
            una de las dietas. Después de un periodo de **90 días**, se registró el peso final (en gramos) 
            de una muestra de truchas de cada poza.
            
            **Objetivo:** Determinar si existen diferencias estadísticamente significativas en la ganancia 
            de peso que puedan ser atribuidas a las diferentes dietas suministradas.
            """)

st.markdown("---")

# Sección 1: Seleccionar Metodología
st.markdown("## 1️⃣ Seleccionar Metodología y Diseño")

# Tabs para diferentes tipos de análisis
tab1, tab2 = st.tabs(["📊 Unifactorial (ANOVA)", "📈 Bifactorial (ANOVA)"])

with tab1:
    st.markdown("### Diseños Unifactoriales")
    
    design_type = st.radio(
        "Seleccione el tipo de diseño:",
        ["unifactorial-dca-balanceado", "unifactorial-dca-no-balanceado"],
        format_func=lambda x: DESIGN_INFO[x]['nombre']
    )
    
    selected_design = design_type
    
    # Mostrar información del diseño
    info = DESIGN_INFO[selected_design]
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Modelo:** {info['modelo']}")
        st.caption(info['descripcion'])
    
    with col2:
        if st.button("📖 Ver Teoría"):
            with st.expander("Teoría del Modelo", expanded=True):
                st.markdown(f"""
                **Modelo Matemático:** {info['modelo']}
                
                **Componentes:**
                - Yij: Observación j del tratamiento i
                - μ: Media general poblacional
                - τi: Efecto del tratamiento i
                - εij: Error experimental ~ N(0, σ²)
                
                **Hipótesis:**
                - H₀: τ₁ = τ₂ = ... = τₖ = 0
                - H₁: Al menos un τi ≠ 0
                """)

with tab2:
    st.markdown("### Diseños Bifactoriales")
    
    design_type = st.radio(
        "Seleccione el tipo de diseño:",
        ["bifactorial-dca-balanceado", "bifactorial-dca-no-balanceado"],
        format_func=lambda x: DESIGN_INFO[x]['nombre'],
        key="bifactorial_radio"
    )
    
    selected_design = design_type
    
    # Mostrar información del diseño
    info = DESIGN_INFO[selected_design]
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Modelo:** {info['modelo']}")
        st.caption(info['descripcion'])
    
    with col2:
        if st.button("📖 Ver Teoría", key="theory_bifactorial"):
            with st.expander("Teoría del Modelo", expanded=True):
                st.markdown(f"""
                **Modelo Matemático:** {info['modelo']}
                
                **Componentes:**
                - Yijk: Observación k de la combinación i,j
                - μ: Media general
                - αi: Efecto del factor A
                - βj: Efecto del factor B
                - (αβ)ij: Interacción A*B
                - εijk: Error experimental
                
                **Hipótesis:**
                - H₀ₐ: α₁ = α₂ = ... = 0
                - H₀ᵦ: β₁ = β₂ = ... = 0
                - H₀ₐᵦ: (αβ)ij = 0
                """)

st.markdown("---")

# Sección 2: Ingresar datos
st.markdown("## 2️⃣ Ingresar y Analizar Datos")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Entrada de Datos")
    
    # Área de texto para ingresar datos
    input_data = st.text_area(
        "Ingrese los datos:",
        value=DEFAULT_DATA.get(selected_design, ''),
        height=300,
        help="Formato: TRATAMIENTO: valor1, valor2, valor3, ..."
    )
    
    if 'unifactorial' in selected_design:
        st.caption("💡 Separe los valores con comas. Cada línea representa un tratamiento.")
    else:
        st.caption("💡 Use formato AxBy: valores (ejemplo: A1B1: 210, 204, 199)")

with col2:
    st.markdown("### 🔬 Análisis")
    st.markdown("Presiona el botón para realizar el análisis estadístico.")
    
    if st.button("🚀 Analizar Datos", type="primary", use_container_width=True):
        try:
            with st.spinner("Procesando datos..."):
                if 'unifactorial' in selected_design:
                    # Análisis unifactorial
                    treatment_names, treatments_data = parse_unifactorial_data(input_data)
                    anova_results = anova_unifactorial(treatments_data)
                    
                    st.session_state['analysis_results'] = {
                        'type': 'unifactorial',
                        'treatment_names': treatment_names,
                        'treatments_data': treatments_data,
                        'anova': anova_results
                    }
                    
                else:
                    # Análisis bifactorial
                    data_map, levels_a, levels_b = parse_bifactorial_data(input_data)
                    anova_results = anova_bifactorial(data_map, levels_a, levels_b)
                    
                    st.session_state['analysis_results'] = {
                        'type': 'bifactorial',
                        'data_map': data_map,
                        'levels_a': levels_a,
                        'levels_b': levels_b,
                        'anova': anova_results
                    }
                
                st.success("✅ Análisis completado exitosamente!")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# Sección 3: Resultados
if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    
    st.markdown("## 3️⃣ Dashboard de Resultados")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>📊 Modelo Estadístico</h4>
            <p style="font-size: 14px; margin: 0;">{DESIGN_INFO[selected_design]['modelo']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if results['type'] == 'unifactorial':
        f_value = results['anova']['fuente'][0]['f']
        p_value = results['anova']['fuente'][0]['p']
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <h4>📈 Estadístico F</h4>
                <h2 style="margin: 0;">{f_value:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = "green" if p_value < 0.05 else "red"
            st.markdown(f"""
            <div class="kpi-card">
                <h4>🎯 P-valor</h4>
                <h2 style="margin: 0; color: {color};">{p_value:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # bifactorial
        p_a = results['anova']['fuente'][0]['p']
        p_b = results['anova']['fuente'][1]['p']
        p_ab = results['anova']['fuente'][2]['p']
        
        with col2:
            color = "green" if p_a < 0.05 else "red"
            st.markdown(f"""
            <div class="kpi-card">
                <h4>P-valor Factor A</h4>
                <h2 style="margin: 0; color: {color};">{p_a:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = "green" if p_b < 0.05 else "red"
            st.markdown(f"""
            <div class="kpi-card">
                <h4>P-valor Factor B</h4>
                <h2 style="margin: 0; color: {color};">{p_b:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Visualización de Datos")
        
        if results['type'] == 'unifactorial':
            # Tabla de datos
            data_dict = {}
            max_len = max(len(d) for d in results['treatments_data'])
            
            for i, (name, data) in enumerate(zip(results['treatment_names'], results['treatments_data'])):
                padded_data = data + [None] * (max_len - len(data))
                data_dict[name] = padded_data
            
            df = pd.DataFrame(data_dict)
            st.dataframe(df, use_container_width=True)
            
        else:  # bifactorial
            # Crear tabla de datos
            rows = []
            for key, values in results['data_map'].items():
                rows.append({
                    'Combinación': key,
                    'Valores': ', '.join(map(str, values)),
                    'n': len(values)
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Gráficos")
        
        if results['type'] == 'unifactorial':
            fig = create_boxplot(results['treatment_names'], results['treatments_data'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_interaction_plot(
                results['data_map'],
                results['levels_a'],
                results['levels_b']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabla ANOVA
    st.markdown("### 📋 Tabla de Análisis de Varianza (ANOVA)")
    
    anova_df = pd.DataFrame(results['anova']['fuente'])
    
    # Formatear valores
    def format_value(val):
        if val == '-':
            return '-'
        try:
            return f"{float(val):.4f}"
        except:
            return str(val)
    
    anova_df['sc'] = anova_df['sc'].apply(format_value)
    anova_df['cm'] = anova_df['cm'].apply(format_value)
    anova_df['f'] = anova_df['f'].apply(format_value)
    anova_df['p'] = anova_df['p'].apply(format_value)
    
    # Renombrar columnas
    anova_df.columns = ['Fuente de Variación', 'GL', 'SC', 'CM', 'F', 'P-valor']
    
    # Mostrar tabla
    st.dataframe(anova_df, use_container_width=True)
    
    # Interpretación
    st.markdown("### 💡 Interpretación (α = 0.05)")
    
    for row in results['anova']['fuente']:
        if isinstance(row['p'], float) and 'total' not in row['source'].lower() and 'error' not in row['source'].lower():
            if row['p'] < 0.05:
                st.success(f"✅ **{row['source']}**: P-valor = {row['p']:.4f}. Se rechaza H₀. Hay efecto significativo.")
            else:
                st.warning(f"⚠️ **{row['source']}**: P-valor = {row['p']:.4f}. No se rechaza H₀. No hay evidencia de efecto significativo.")

else:
    st.info("👆 Ingrese los datos y presione 'Analizar Datos' para ver los resultados.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Dashboard de Diseños Experimentales | Universidad Nacional del Altiplano - Puno</p>
    <p style="font-size: 12px;">Desarrollado para el curso de Diseños Experimentales 2025-II</p>
    <p style="font-size: 12px; margin-top: 10px;">
        <a href="https://github.com/tuusuario/turepositorio" target="_blank">⭐ Ver en GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)