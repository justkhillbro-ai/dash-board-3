import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import erfc

# ==========================================
# 0. CONFIGURAZIONE PAGINA E COSTANTI
# ==========================================
st.set_page_config(page_title="Nokia FD - Caso 2 (Hardware Reale)", layout="wide")

# Costanti di sistema (Nokia Baseline)
fc = 145e9      # 145 GHz
c = 3e8         # m/s
Gtx_dBi = 40    # dBi
Grx_dBi = 40    # dBi
T0 = 290        # K
k_B = 1.38e-23  # J/K
M = 16          # 16-QAM per la simulazione del clipping

# ==========================================
# 1. SIDEBAR: CONTROLLI INTERATTIVI
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Nokia_wordmark.svg/1200px-Nokia_wordmark.svg.png", width=150)
st.sidebar.header("⚙️ Parametri Operativi")

d_m = st.sidebar.slider("Distanza Link (m)", 100, 1000, 500, step=50)
iso_dB = st.sidebar.slider("Isolamento Antenna (dB)", 40, 80, 55, step=1)
B_MHz = st.sidebar.slider("Banda del Canale (MHz)", 250, 2000, 1000, step=50)
NF_dB = st.sidebar.slider("Noise Figure RX (dB)", 4, 10, 6, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Parametri Amplificatore (PA)")
Psat_dBm = st.sidebar.slider("Potenza Saturazione (Psat) [dBm]", 0, 30, 15, step=1)
IBO_dB = st.sidebar.slider("Input Back-Off (IBO) [dB]", 0, 15, 5, step=1)

# Calcoli base: La Potenza TX dipende dall'IBO impostato
Ptx_dBm = Psat_dBm - IBO_dB
Ptx_W = 10 ** ((Ptx_dBm - 30) / 10)
B_Hz = B_MHz * 1e6

# Link Budget
FSPL_dB = 20 * np.log10(d_m) + 20 * np.log10(fc) + 20 * np.log10(4 * np.pi / c)
Prx_dBm = Ptx_dBm + Gtx_dBi + Grx_dBi - FSPL_dB
Prx_W = 10 ** ((Prx_dBm - 30) / 10)

Pn_W = k_B * T0 * B_Hz * (10 ** (NF_dB / 10))
Pi_W = 10 ** ((Ptx_dBm - iso_dB - 30) / 10)

# ==========================================
# SIMULAZIONE VELOCE DISTORSIONE PA
# ==========================================
n_sym = 5000
np.random.seed(42) # Fissiamo il seed per evitare sfarfallii nei grafici
I_ideal = np.random.choice([-3, -1, 1, 3], n_sym)
Q_ideal = np.random.choice([-3, -1, 1, 3], n_sym)
sym_tx = (I_ideal + 1j*Q_ideal)
sym_tx_norm = sym_tx / np.sqrt(np.mean(np.abs(sym_tx)**2))

# Funzione per calcolare la varianza di distorsione normalizzata in base all'IBO
def calc_dist_var(ibo):
    A_sat = 10 ** (-ibo / 20) 
    amp = np.abs(sym_tx_norm)
    clip_mask = amp > A_sat
    sym_clip = np.copy(sym_tx_norm)
    sym_clip[clip_mask] = sym_tx_norm[clip_mask] * (A_sat / amp[clip_mask])
    return np.mean(np.abs(sym_tx_norm - sym_clip)**2), sym_clip

P_dist_norm, sym_clip_current = calc_dist_var(IBO_dB)
P_dist_rx_W = P_dist_norm * Prx_W # La distorsione viaggia con il segnale utile

# ==========================================
# HEADER NARRATIVO
# ==========================================
st.title("🔋 Caso 2: Limiti Hardware e PA Clipping (Simmetrico)")
st.markdown("Questa dashboard analizza l'impatto delle **non-linearità dell'amplificatore di potenza (PA)**. Entrambi i nodi lavorano alla stessa potenza e subiscono saturazione. Il parametro chiave diventa l'**Input Back-Off (IBO)**: occorre trovare il compromesso tra potenza trasmessa e distorsione del segnale.")
st.markdown("---")

# ==========================================
# SEZIONE 1: L'impatto fisico (Costellazione)
# ==========================================
st.header("1. Effetto della Non-Linearità sulla Costellazione")
st.markdown("A differenza del Caso 1 (dove la nuvola era circolare per via del rumore termico), il clipping del PA **schiaccia e comprime** i punti esterni della costellazione M-QAM, limitandone l'ampiezza massima.")

rx_points = sym_clip_current * np.sqrt(Prx_W)
n_t = (np.random.randn(n_sym) + 1j*np.random.randn(n_sym)) * np.sqrt((Pn_W + Pi_W)/2)
rx_points_tot = rx_points + n_t

fig1 = go.Figure()
plot_points = rx_points_tot / np.sqrt(Prx_W) # Normalizziamo per la visualizzazione
fig1.add_trace(go.Scatter(x=plot_points.real, y=plot_points.imag, mode='markers', marker=dict(size=4, color='rgba(255,127,14,0.6)'), name="Ricevuti (Clipping + Interferenza)"))

I_ref = np.array([-3, -1, 1, 3]) / np.sqrt(10)
Q_ref = np.array([-3, -1, 1, 3]) / np.sqrt(10)
ref_pts = [complex(i, q) for i in I_ref for q in Q_ref]
fig1.add_trace(go.Scatter(x=[p.real for p in ref_pts], y=[p.imag for p in ref_pts], mode='markers', marker=dict(symbol='cross', size=12, color='black'), name="Ideali di Riferimento"))

fig1.update_layout(title=f"Costellazione 16-QAM | IBO: {IBO_dB} dB | Distanza: {d_m} m", xaxis_title="I", yaxis_title="Q", height=550, template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ==========================================
# CALCOLI VETTORIALI PER GRAFICI 2 E 3
# ==========================================
ibo_vec = np.linspace(0, 15, 40)
evm_vec = []
cap_fd_vec = []

for ibo_val in ibo_vec:
    ptx_temp = Psat_dBm - ibo_val
    prx_w_temp = 10 ** ((ptx_temp + Gtx_dBi + Grx_dBi - FSPL_dB - 30) / 10)
    pi_w_temp = 10 ** ((ptx_temp - iso_dB - 30) / 10)
    
    dist_norm_temp, _ = calc_dist_var(ibo_val)
    p_dist_rx_temp = dist_norm_temp * prx_w_temp
    
    # EVM = sqrt((Rumore + Interferenza + Distorsione_RX) / Potenza_Segnale_RX)
    evm = np.sqrt((Pn_W + pi_w_temp + p_dist_rx_temp) / prx_w_temp) * 100
    evm_vec.append(evm)
    
    # Capacità con SNDR (Signal to Noise, Distortion and Interference Ratio)
    sndr = prx_w_temp / (Pn_W + pi_w_temp + p_dist_rx_temp)
    cap_fd_vec.append((B_Hz * np.log2(1 + sndr)) / 1e9)

# ==========================================
# SEZIONE 2: EVM vs IBO
# ==========================================
st.header("2. Metrica Industriale: EVM vs Input Back-Off")
st.markdown("L'EVM (Error Vector Magnitude) quantifica il degrado totale del sistema. Riducendo l'IBO, ci si avvicina alla potenza di saturazione dell'amplificatore: la distorsione aumenta drasticamente causando un'impennata dell'EVM.")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ibo_vec, y=evm_vec, mode='lines', line=dict(color='red', width=3), name="EVM (%)"))
fig2.add_vline(x=IBO_dB, line=dict(color="gray", dash="dot"), annotation_text="Il tuo IBO attuale")
fig2.update_layout(xaxis_title="Input Back-Off (IBO) [dB]", yaxis_title="EVM (%)", yaxis_type="log", height=500, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 3: Capacità vs IBO (Trade-off)
# ==========================================
st.header("3. Il Trade-off Ingegneristico: Ottimizzazione della Capacità")
st.markdown("Questo è il grafico fondamentale per il progettista hardware. A sinistra (IBO basso) la capacità crolla a causa della forte distorsione introdotta dal clipping. A destra (IBO alto) l'amplificatore è lineare, ma la potenza trasmessa è troppo bassa e il segnale viene sommerso dal rumore termico.")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=ibo_vec, y=cap_fd_vec, mode='lines', line=dict(color='green', width=3), name="Capacità Full-Duplex"))
fig3.add_vline(x=IBO_dB, line=dict(color="gray", dash="dot"), annotation_text="Punto Operativo")

opt_ibo = ibo_vec[np.argmax(cap_fd_vec)]
fig3.add_vline(x=opt_ibo, line=dict(color="blue", dash="dash"), annotation_text=f"Ottimo Teorico: {opt_ibo:.1f} dB")

fig3.update_layout(xaxis_title="Input Back-Off (IBO) [dB]", yaxis_title="Capacità Netta (Gbps)", height=500, template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 4: Capacità vs Isolamento (Ideale vs Reale)
# ==========================================
st.header("4. Degrado di Sistema: Modello Ideale vs Hardware Reale")
st.markdown("Confrontiamo le prestazioni del sistema mantenendo l'IBO scelto nella sidebar. La curva tratteggiata rappresenta il caso ideale (PA perfettamente lineare descritto nel Caso 1), mentre la curva solida include la perdita dovuta all'hardware fisico commerciale.")

iso_array = np.linspace(40, 80, 50)
cap_ideal = []
cap_real = []

for iso_val in iso_array:
    pi_val = 10 ** ((Ptx_dBm - iso_val - 30) / 10)
    # Ideale: No distorsione hardware
    cap_ideal.append((B_Hz * np.log2(1 + Prx_W/(Pn_W + pi_val))) / 1e9)
    # Reale: Con distorsione clipping fissata per l'attuale IBO
    cap_real.append((B_Hz * np.log2(1 + Prx_W/(Pn_W + pi_val + P_dist_rx_W))) / 1e9)
    
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=iso_array, y=cap_ideal, mode='lines', name="Caso 1: Modello Ideale", line=dict(color='black', dash='dash')))
fig4.add_trace(go.Scatter(x=iso_array, y=cap_real, mode='lines', name=f"Caso 2: Reale (IBO = {IBO_dB} dB)", line=dict(color='blue', width=3)))
fig4.update_layout(xaxis_title="Isolamento Antenna (dB)", yaxis_title="Capacità di Canale (Gbps)", height=500, template="plotly_white")
fig4.add_vline(x=iso_dB, line=dict(color="gray", dash="dot"), annotation_text="Isolamento Attuale")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 5: La Visione Commerciale (Heatmap IBO-Isolamento)
# ==========================================
st.header("5. Mappa Strategica di Co-Design (Isolamento RF vs IBO Baseband)")
st.markdown("Questa Heatmap unisce le competenze di due team ingegneristici in Nokia. Per massimizzare la capacità del link (zone gialle/chiare), occorre ottimizzare simultaneamente l'isolamento garantito dalle antenne (Asse Y) e la calibrazione del Back-Off dell'amplificatore (Asse X).")

Ibo_mesh, Iso_mesh = np.meshgrid(np.linspace(0, 15, 30), np.linspace(40, 80, 30))

# Vettorizziamo la funzione di calcolo varianza distorsione per la heatmap per evitare lag
dist_vars_interp = np.array([calc_dist_var(i)[0] for i in np.linspace(0, 15, 30)])
P_dist_norm_mesh = np.interp(Ibo_mesh, np.linspace(0, 15, 30), dist_vars_interp)

Ptx_mesh_W = 10 ** ((Psat_dBm - Ibo_mesh - 30) / 10)
Prx_mesh_W = Ptx_mesh_W * (10 ** ((Gtx_dBi + Grx_dBi - FSPL_dB) / 10))
Pi_mesh_W = Ptx_mesh_W * (10 ** (-Iso_mesh / 10))
P_dist_mesh_W = P_dist_norm_mesh * Prx_mesh_W

SNDR_mesh = Prx_mesh_W / (Pn_W + Pi_mesh_W + P_dist_mesh_W)
Cap_mesh = (B_Hz * np.log2(1 + SNDR_mesh)) / 1e9

fig5 = go.Figure(data=go.Heatmap(
    z=Cap_mesh, 
    x=np.linspace(0, 15, 30), 
    y=np.linspace(40, 80, 30), 
    colorscale='Viridis', 
    colorbar=dict(title="Capacità (Gbps)")
))
fig5.update_layout(xaxis_title="Input Back-Off Amplificatore (IBO) [dB]", yaxis_title="Isolamento Antenna (dB)", height=600, template="plotly_white")

# Marker per identificare il punto di lavoro attuale
fig5.add_trace(go.Scatter(x=[IBO_dB], y=[iso_dB], mode='markers+text', text=["Punto Operativo"], marker=dict(size=12, color='white', symbol='diamond'), textposition="top center", showlegend=False))

st.plotly_chart(fig5, use_container_width=True)