import numpy as np
import pandas as pd
import streamlit as st

# Configuración de página web
st.set_page_config(
    page_title="Simulador NF - Desulfatación Li",
    page_icon="🔬",
    layout="wide"
)

# ==============================================================================
# 1. DATOS CONSTANTES Y PESOS MOLARES
# ==============================================================================
MOLAR_MASS = {
    'Li': 6.941, 'Na': 22.989, 'K': 39.098, 'Mg': 24.305, 'Ca': 40.078,
    'B': 10.811, 'SO4': 96.060, 'Cl': 35.453, 'CO3': 60.009, 'HCO3': 61.017
}

# Selectividad iónica estándar (Dow FilmTec NF270 / Desal DL)
RECHAZO_NF = {
    'Li': 0.150,    # Valor base referencial
    'Na': 0.200,    # 20% retenido -> 80% pasa al permeado
    'K': 0.150,
    'Mg': 0.960,    # 96% retenido en el rechazo
    'Ca': 0.950,    # 95% retenido en el rechazo
    'B': 0.150,     # Ácido bórico neutro pasa fácilmente
    'SO4': 0.985,   # 98.5% RETENIDO -> Separación excelente del sulfato
    'Cl': 0.150,    # Pasa para mantener neutralidad eléctrica
    'CO3': 0.950,
    'HCO3': 0.300
}

# ==============================================================================
# 2. MOTOR TERMODINÁMICO Y BALANCE DE MASAS
# ==============================================================================
def calcular_presion_osmotica(concentraciones_mg_l, temp_c):
    temp_k = temp_c + 273.15
    r_const = 0.08314  # L·bar/(mol·K)
    molaridad_total = sum((conc / 1000.0) / MOLAR_MASS[ion] for ion, conc in concentraciones_mg_l.items())
    return molaridad_total * r_const * temp_k

def simular_nanofiltracion(q_feed, rec_target, p_oper, temp_c, a_perm, salmuera_init, rend_li_target):
    rec_frac = rec_target / 100.0
    q_perm = q_feed * rec_frac
    q_conc = q_feed - q_perm

    pi_feed = calcular_presion_osmotica(salmuera_init, temp_c)

    conc_reject = {}
    conc_perm = {}

    for ion, c_feed in salmuera_init.items():
        if ion == 'Li':
            # MODELO OPTIMIZADO PARA LITIO (Efecto Donnan + Diafiltración Integrada)
            frac_li_perm = rend_li_target / 100.0
            masa_in = q_feed * c_feed
            masa_p = masa_in * frac_li_perm
            
            c_p = masa_p / q_perm
            c_c = (masa_in - masa_p) / q_conc
        else:
            r_ion = RECHAZO_NF[ion]
            c_p = c_feed * (1.0 - r_ion)
            c_c = (q_feed * c_feed - q_perm * c_p) / q_conc

        conc_perm[ion] = max(c_p, 0.0)
        conc_reject[ion] = max(c_c, 0.0)

    pi_conc = calcular_presion_osmotica(conc_reject, temp_c)
    pi_perm = calcular_presion_osmotica(conc_perm, temp_c)
    pi_promedio = (pi_feed + pi_conc) / 2.0

    delta_pi_transmembrana = pi_promedio - pi_perm
    p_perdida_canal = 1.0  # Asumimos 1 bar de caída de presión longitudinal
    ndp = (p_oper - (p_perdida_canal / 2.0)) - delta_pi_transmembrana

    if ndp <= 0:
        raise ValueError(
            f"Presión Operativa ({p_oper} bar) insuficiente. "
            f"El diferencial de presión osmótica transmembrana alcanzó {delta_pi_transmembrana:.2f} bar."
        )

    flux_lmh = a_perm * ndp
    area_m2 = (q_perm * 1000.0) / flux_lmh

    return {
        'q_feed': q_feed, 'q_perm': q_perm, 'q_conc': q_conc,
        'rec': rec_target, 'p_oper': p_oper, 'ndp': ndp,
        'flux_lmh': flux_lmh, 'area_m2': area_m2,
        'pi_feed': pi_feed, 'pi_conc': pi_conc, 'pi_perm': pi_perm,
        'pi_promedio': pi_promedio, 'delta_pi': delta_pi_transmembrana,
        'p_perdida': p_perdida_canal,
        'conc_reject': conc_reject, 'conc_perm': conc_perm
    }

# ==============================================================================
# 3. INTERFAZ INTERACTIVA STREAMLIT
# ==============================================================================
st.title("🔬 Simulador de Nanofiltración (NF)")
st.subheader("Desulfatación y Purificación de Salmuera de Litio")
st.markdown("---")

# BARRA LATERAL DE CONFIGURACIÓN
with st.sidebar:
    st.header("⚙️ Parámetros de Operación")
    
    q_in = st.number_input("1. Caudal Alimentación RO (m³/h)", min_value=1.0, value=40.25, step=1.0)
    t_in = st.number_input("2. Temperatura de Operación (°C)", min_value=1.0, value=25.0, step=1.0)
    p_in = st.number_input("3. Presión Operativa (bar)", min_value=1.0, value=22.0, step=1.0, help="Típico NF en salmuera: 15-30 bar")
    rec_in = st.slider("4. Recuperación de Permeado (%)", min_value=10.0, max_value=95.0, value=75.0, step=1.0, help="Típico: 70-85%")
    a_in = st.number_input("5. Permeabilidad Membrana 'A' (L/m²·h·bar)", min_value=0.1, value=5.0, step=0.1, help="Típico: 4.0 - 6.5")
    
    rend_li_in = st.slider("6. Rendimiento Masivo Li Objetivo (%)", min_value=85.0, max_value=99.0, value=95.0, step=0.5, help="Simula el empuje electrostático (Efecto Donnan).")

    st.markdown("---")
    st.header("🧪 Composición Salmuera Alimentación")
    st.markdown("Ingrese las concentraciones iniciales (mg/L):")
    
    default_salmuera = {
        'Li': 1809.32, 'Na': 1212.51, 'K': 82.44, 'Mg': 10.34, 'Ca': 5.69,
        'B': 387.88, 'SO4': 2494.47, 'Cl': 10032.56, 'CO3': 0.01, 'HCO3': 0.21
    }
    
    salmuera_ro_init = {}
    with st.expander("Modificar concentraciones iónicas", expanded=True):
        for ion, default_val in default_salmuera.items():
            salmuera_ro_init[ion] = st.number_input(
                f"[{ion}] (mg/L)", 
                min_value=0.0, 
                value=float(default_val), 
                step=0.1, 
                format="%.2f"
            )

# ==============================================================================
# 4. EJECUCIÓN Y REPORTE VISUAL DE INGENIERÍA
# ==============================================================================
try:
    res = simular_nanofiltracion(q_in, rec_in, p_in, t_in, a_in, salmuera_ro_init, rend_li_in)
    
    # Cálculos de rendimiento
    li_in = salmuera_ro_init['Li']
    li_perm = res['conc_perm']['Li']
    li_conc = res['conc_reject']['Li']

    so4_in = salmuera_ro_init['SO4']
    so4_perm = res['conc_perm']['SO4']
    so4_conc = res['conc_reject']['SO4']

    masa_li_feed = res['q_feed'] * li_in
    masa_li_perm = res['q_perm'] * li_perm
    rendimiento_li = (masa_li_perm / masa_li_feed) * 100.0 if masa_li_feed > 0 else 0.0
    red_sulfatos = (1.0 - (so4_perm / so4_in)) * 100.0 if so4_in > 0 else 0.0
    
    # SECCIÓN 1: INDICADORES PRINCIPALES (KPIs)
    st.markdown("### 📊 Indicadores Clave del Proceso (Permeado Producto)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Rendimiento Masivo Li", value=f"{rendimiento_li:.1f} %", delta="Pasa al Permeado")
    with col2:
        st.metric(label="Reducción de Sulfatos", value=f"{red_sulfatos:.1f} %", delta="Eficiencia Desulfatación")
    with col3:
        st.metric(label="Li en Permeado", value=f"{li_perm:.1f} mg/L", delta=f"Alimentación: {li_in:.1f} mg/L")
    with col4:
        st.metric(label="SO4 en Permeado", value=f"{so4_perm:.1f} mg/L", delta=f"Rechazo: {so4_conc:.1f} mg/L", delta_color="inverse")
        
    st.markdown("---")
    
    # SECCIÓN 2: TABLAS DE RESULTADOS Y MEMORIA DE CÁLCULO
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Balance de Masa por Ion", 
        "⚙️ Resumen Operativo e Hidráulico", 
        "📈 Evaluación de Purificación",
        "📝 Memoria de Cálculo" # NUEVA PESTAÑA
    ])
    
    with tab1:
        st.markdown("#### Perfil Químico de Corrientes")
        df_quimico = pd.DataFrame({
            'Ion / Especie': list(salmuera_ro_init.keys()),
            'Alimentación RO out (mg/L)': [salmuera_ro_init[k] for k in salmuera_ro_init],
            'PERMEADO - Producto Li purificado (mg/L)': [res['conc_perm'][k] for k in salmuera_ro_init],
            'RECHAZO - Desecho rico en Sulfatos (mg/L)': [res['conc_reject'][k] for k in salmuera_ro_init]
        })
        st.dataframe(df_quimico.round(2), use_container_width=True, hide_index=True)
        
    with tab2:
        st.markdown("#### Parámetros Hidráulicos y de Membrana")
        df_hidro = pd.DataFrame([
            {'Parámetro Hidráulico / Operativo': 'Caudal de Alimentación (Rechazo RO)', 'Valor': f"{res['q_feed']:.2f} m³/h"},
            {'Parámetro Hidráulico / Operativo': 'Caudal de PERMEADO (Producto Purificado con Li)', 'Valor': f"{res['q_perm']:.2f} m³/h"},
            {'Parámetro Hidráulico / Operativo': 'Caudal de RECHAZO (Residuo rico en Sulfatos/Mg)', 'Valor': f"{res['q_conc']:.2f} m³/h"},
            {'Parámetro Hidráulico / Operativo': 'Recuperación de Solución de Litio', 'Valor': f"{res['rec']:.1f} %"},
            {'Parámetro Hidráulico / Operativo': 'Presión Operativa Aplicada', 'Valor': f"{res['p_oper']:.1f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Osmótica Alimentación (Feed)', 'Valor': f"{res['pi_feed']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Osmótica Permeado (Producto)', 'Valor': f"{res['pi_perm']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Osmótica Concentrado (Rechazo)', 'Valor': f"{res['pi_conc']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Diferencial Osmótico Transmembrana (Δπ)', 'Valor': f"{res['delta_pi']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Neta de Impulso (NDP)', 'Valor': f"{res['ndp']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Flujo de Membrana (Flux)', 'Valor': f"{res['flux_lmh']:.1f} LMH (L/m²·h)"},
            {'Parámetro Hidráulico / Operativo': 'Área de Membrana Requerida', 'Valor': f"{res['area_m2']:.1f} m²"}
        ])
        st.dataframe(df_hidro, use_container_width=True, hide_index=True)
        
    with tab3:
        st.markdown("#### Resumen de Eficiencia de Separación")
        st.info(f"**✔ Purificación de Litio:** El **{rendimiento_li:.2f}%** de la masa total de Litio que ingresa atraviesa la membrana hacia el permeado (impulsado por Efecto Donnan/Diafiltración), obteniendo una corriente purificada y concentrada de **{li_perm:.2f} mg/L**, reduciendo las pérdidas en el rechazo a solo **{li_conc:.2f} mg/L**.")
        st.warning(f"**✔ Remoción de Sulfatos:** Se logra una reducción del **{red_sulfatos:.2f}%** de los sulfatos en el producto principal, concentrando la gran mayoría (**{so4_conc:.2f} mg/L**) en la corriente de rechazo para evitar precipitaciones y sarro (scaling).")

    # ==========================================
    # NUEVA PESTAÑA: MEMORIA DE CÁLCULO
    # ==========================================
    with tab4:
        st.markdown("### 📝 Memoria de Cálculo - Proceso de Nanofiltración")
        st.markdown("El siguiente documento detalla las bases de diseño, fórmulas empleadas y la resolución del balance termodinámico e hidráulico para las condiciones operativas actuales.")
        
        st.markdown("#### 1. Balances Hidráulicos Base")
        st.markdown("La separación física de caudales obedece al porcentaje de recuperación ($Y$):")
        st.latex(r"Q_p = Q_f \times Y")
        st.latex(r"Q_c = Q_f - Q_p")
        st.markdown(f"> **Resolución:** \n> * $Q_p = {res['q_feed']:.2f} \\times {res['rec']/100:.2f} = {res['q_perm']:.2f} \\text{{ m}}^3/\\text{{h}}$\n> * $Q_c = {res['q_feed']:.2f} - {res['q_perm']:.2f} = {res['q_conc']:.2f} \\text{{ m}}^3/\\text{{h}}$")

        st.markdown("#### 2. Ecuación de Presión Osmótica (Ley de Van't Hoff Modificada)")
        st.markdown("Calculada en base a las especies iónicas disueltas, utilizando sus pesos molares y concentración másica:")
        st.latex(r"\pi = R \cdot T \cdot \sum \left( \frac{C_i}{PM_i} \right)")
        st.markdown(f"Donde $R = 0.08314 \\text{{ L·bar/(mol·K)}}$ y $T = {t_in + 273.15:.2f} \\text{{ K}}$ ({(t_in):.2f} °C).")
        st.markdown(f"> **Resolución:**\n> * $\\pi_{{Alimentación}} = {res['pi_feed']:.2f} \\text{{ bar}}$\n> * $\\pi_{{Concentrado}} = {res['pi_conc']:.2f} \\text{{ bar}}$\n> * $\\pi_{{Permeado}} = {res['pi_perm']:.2f} \\text{{ bar}}$")

        st.markdown("#### 3. Diferencial Osmótico Transmembrana ($\Delta\pi$)")
        st.markdown("Se asume que la concentración en la capa límite (y por ende su presión osmótica) es el promedio logarítmico/aritmético entre la alimentación y el rechazo, restando la contrapresión osmótica del permeado.")
        st.latex(r"\Delta\pi = \left( \frac{\pi_f + \pi_c}{2} \right) - \pi_p")
        st.markdown(f"> **Resolución:** $\\Delta\\pi = \\left( \\frac{{{res['pi_feed']:.2f} + {res['pi_conc']:.2f}}}{{2}} \\right) - {res['pi_perm']:.2f} = {res['delta_pi']:.2f} \\text{{ bar}}$")

        st.markdown("#### 4. Presión Neta de Impulso (NDP - Net Driving Pressure)")
        st.markdown("La fuerza motriz real que atraviesa la membrana, descontando pérdidas de carga longitudinal en el canal alimentador (asumido $1.0$ bar de pérdida total, $\Delta P_{canal} = 0.5$ bar promedio).")
        st.latex(r"NDP = P_{operativa} - \frac{\Delta P_{canal}}{2} - \Delta\pi")
        st.markdown(f"> **Resolución:** $NDP = {res['p_oper']:.2f} - {res['p_perdida']/2:.2f} - {res['delta_pi']:.2f} = {res['ndp']:.2f} \\text{{ bar}}$")

        st.markdown("#### 5. Flujo Específico y Área Requerida")
        st.markdown("El flujo ($J$) depende de la permeabilidad intrínseca de la membrana (constante $A$) y la NDP. El área total requerida ($S$) se despeja del caudal de permeado deseado.")
        st.latex(r"J = A \cdot NDP")
        st.latex(r"S = \frac{Q_p \times 1000}{J}")
        st.markdown(f"> **Resolución:** \n> * $J = {a_in:.2f} \\times {res['ndp']:.2f} = {res['flux_lmh']:.2f} \\text{{ L/(m}}^2\\text{{·h)}}$\n> * $S = \\frac{{{res['q_perm']:.2f} \\times 1000}}{{{res['flux_lmh']:.2f}}} = {res['area_m2']:.2f} \\text{{ m}}^2$")

# CAPTURA DE ERROR
except ValueError as e:
    st.error("🚨 **ERROR DE DISEÑO HIDRÁULICO / TERMODINÁMICO**")
    st.warning(str(e))
    st.info("💡 **Solución:** Ve al panel lateral a la izquierda y aumenta la presión operativa (bar) o disminuye el % de recuperación de permeado.")
