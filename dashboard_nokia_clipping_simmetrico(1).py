import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import erfc

# ==========================================
# 0. CONFIGURAZIONE PAGINA E COSTANTI
# ==========================================
st.set_page_config(page_title="Nokia FD - Case 2 (PA Clipping))", layout="wide")

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
st.sidebar.header("Netork Parameters")

d_m = st.sidebar.slider("Link Distance (m)", 100, 1000, 500, step=50)
iso_dB = st.sidebar.slider("Antenna Isolation (dB)", 40, 80, 55, step=1)
B_MHz = st.sidebar.slider("Channel Bandwidth (MHz)", 250, 2000, 1000, step=50)
NF_dB = st.sidebar.slider("RX Noise Figure (dB)", 4, 10, 6, step=1)

st.sidebar.markdown("---")

st.sidebar.header("RF Amplifier (PA) Parameters")
Psat_dBm = st.sidebar.slider("Saturation Power (Psat) [dBm]", 0, 30, 15, step=1)
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
# HEADER NARRATIVO (Titolo visibile)
# ==========================================
st.title("D-Band Full-Duplex: Nonlinear Hardware")
st.markdown("**Case 2: Symmetric PA Clipping**. Introducing Power Amplifier (PA) non-linearities. A low Input Back-Off (IBO) pushes the amplifier near saturation, increasing TX power but generating amplitude clipping (distortion). A high IBO yields a clean signal but drastically reduces radiated power.")

# La formula del SNDR
st.latex(r"C_{real} = B \log_2\left(1 + \frac{P_{RX}}{P_{noise} + P_{leakage} + \mathbf{P_{distortion}}}\right)")

st.markdown("---")

# ==========================================
# SIMULAZIONE VELOCE DISTORSIONE PA
# ==========================================
n_sym = 5000
np.random.seed(42) # Fissiamo il seed per evitare sfarfallii nei grafici
I_ideal = np.random.choice([-3, -1, 1, 3], n_sym) / np.sqrt(10)
Q_ideal = np.random.choice([-3, -1, 1, 3], n_sym) / np.sqrt(10)
sym_tx_norm = (I_ideal + 1j*Q_ideal)

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
# SEZIONE 1: L'impatto fisico (Costellazione)
# ==========================================
st.header("1. Non-Linearity Impact on the Constellation")
st.markdown("Unlike the purely additive Gaussian noise seen in Case 1, the Power Amplifier's hard clipping **radially compresses** the outermost symbols of the M-QAM constellation. When the IBO is low, the signal hits the saturation limit ($A_{sat}$), creating a distinct circular boundary and destroying amplitude information.")

# Calcolo punti ricevuti
rx_points = sym_clip_current * np.sqrt(Prx_W)
n_t = (np.random.randn(n_sym) + 1j*np.random.randn(n_sym)) * np.sqrt((Pn_W + Pi_W)/2)
rx_points_tot = rx_points + n_t

fig1 = go.Figure()
plot_points = rx_points_tot / np.sqrt(Prx_W) # Normalizziamo per la visualizzazione

fig1.add_trace(go.Scatter(x=plot_points.real, y=plot_points.imag, mode='markers', 
                          marker=dict(size=3, color='rgba(0, 51, 102, 0.5)'), 
                          name="Received (Clipping + SI + AWGN)"))

ref_pts = [complex(i, q) for i in (np.array([-3, -1, 1, 3]) / np.sqrt(10)) for q in (np.array([-3, -1, 1, 3]) / np.sqrt(10))]
fig1.add_trace(go.Scatter(x=[p.real for p in ref_pts], y=[p.imag for p in ref_pts], mode='markers', 
                          marker=dict(symbol='cross', size=12, color='red'), 
                          name="Ideal Reference"))
fig1.update_layout(
    title=f"16-QAM Constellation | IBO: {IBO_dB} dB | Distance: {d_m} m",
    xaxis_title="In-Phase (I)",
    yaxis_title="Quadrature (Q)",
    height=600, 
    template="plotly_white",
    yaxis=dict(scaleanchor="x", scaleratio=1, zeroline=True, zerolinewidth=1.5, zerolinecolor='black'),
    xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='black')
)

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
st.header("2. Industrial Metric: EVM vs. Input Back-Off")
st.markdown("The **Error Vector Magnitude (EVM)** quantifies the total system degradation. Driving the Power Amplifier with a low IBO pushes it near saturation: non-linear distortion spikes, causing a rapid deterioration of the EVM. A high EVM means the receiver cannot reliably map the QAM symbols.")

col_evm1, col_evm2 = st.columns(2)
with col_evm1:
    st.markdown("**Statistical Definition:**")
    st.latex(r"EVM_{RMS}(\%) = \sqrt{\frac{\frac{1}{N} \sum_{k=1}^{N} |S_{rx,k} - S_{ideal,k}|^2}{P_{avg}}} \times 100")
with col_evm2:
    st.markdown("**System Approximation:**")
    st.latex(r"EVM \approx \frac{1}{\sqrt{SNDR}}")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=ibo_vec, 
    y=evm_vec, 
    mode='lines', 
    line=dict(color='#d62728', width=3), # Rosso per indicare degrado
    name="EVM",
    hovertemplate="IBO: %{x:.1f} dB<br>EVM: %{y:.2f}%<extra></extra>"
))

fig2.update_layout(
    xaxis_title="Input Back-Off (IBO) [dB]", 
    yaxis_title="Error Vector Magnitude (EVM) [%]", 
    yaxis_type="log",
    yaxis=dict(ticksuffix="%"),
    height=500, 
    template="plotly_white",
    hovermode="x unified"
)
fig2.add_vline(x=IBO_dB, line=dict(color="gray", dash="dot"), annotation_text="Current IBO", annotation_position="top right")

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 3: Capacità vs IBO (Trade-off)
# ==========================================
st.header("3. The Engineering Trade-off: Capacity Optimization")
st.markdown("This is the core architectural chart for the RF designer. On the left (**low IBO**), capacity collapses due to severe non-linear distortion (clipping). On the right (**high IBO**), the amplifier operates in its linear region, but the radiated power is too low, causing the signal to drown in the thermal noise floor. The peak of the curve represents the **optimal operating point**.")

fig3 = go.Figure()

# Tracciato della curva di Capacità
fig3.add_trace(go.Scatter(
    x=ibo_vec, 
    y=cap_fd_vec, 
    mode='lines', 
    line=dict(color='#2ca02c', width=3), # Verde semantico per la capacità
    name="Net Capacity",
    hovertemplate="IBO: %{x:.1f} dB<br>Capacity: %{y:.2f} Gbps<extra></extra>"
))

# Calcolo del punto ottimo
opt_ibo = ibo_vec[np.argmax(cap_fd_vec)]
opt_cap = np.max(cap_fd_vec)

# Marker per il picco massimo (Stella blu)
fig3.add_trace(go.Scatter(
    x=[opt_ibo], 
    y=[opt_cap], 
    mode='markers', 
    marker=dict(color='#1f77b4', size=12, symbol='star'), 
    showlegend=False, 
    hoverinfo='skip'
))

# Linea per l'IBO ottimale
fig3.add_vline(x=opt_ibo, line=dict(color="#1f77b4", dash="dash"), annotation_text=f"Optimal IBO: {opt_ibo:.1f} dB", annotation_position="top left")

# Linea per l'IBO attuale impostato sulla sidebar
fig3.add_vline(x=IBO_dB, line=dict(color="gray", dash="dot"), annotation_text="Current IBO", annotation_position="bottom right")

fig3.update_layout(
    xaxis_title="Input Back-Off (IBO) [dB]", 
    yaxis_title="Net Capacity (Gbps)", 
    height=500, 
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 4: Capacità vs Isolamento (Ideale vs Reale)
# ==========================================
st.header("4. System Degradation: Ideal vs. Real Hardware")
st.markdown("This chart bridges the gap between theoretical limits and actual deployment. We compare the system's performance at the current IBO setting against the ideal baseline (Case 1). The vertical gap between the two curves represents the **Distortion Penalty**—the pure capacity lost due to the amplifier's non-linearities.")

iso_array = np.linspace(40, 80, 50)

# Vettorializzazione dei calcoli (rimosso il ciclo for per massime prestazioni)
pi_array = 10 ** ((Ptx_dBm - iso_array - 30) / 10)

# Calcolo Capacità Ideale (Senza distorsione)
cap_ideal = (B_Hz * np.log2(1 + Prx_W / (Pn_W + pi_array))) / 1e9

# Calcolo Capacità Reale (Con distorsione fissata per l'attuale IBO)
cap_real = (B_Hz * np.log2(1 + Prx_W / (Pn_W + pi_array + P_dist_rx_W))) / 1e9

fig4 = go.Figure()

# Traccia Ideale (Nera tratteggiata come riferimento teorico)
fig4.add_trace(go.Scatter(
    x=iso_array, 
    y=cap_ideal, 
    mode='lines', 
    name="Case 1 (Ideal PA)", 
    line=dict(color='black', dash='dash', width=2),
    hovertemplate="Ideal Capacity: %{y:.2f} Gbps<extra></extra>"
))

# Traccia Reale (Blu Unipi, il focus del grafico)
fig4.add_trace(go.Scatter(
    x=iso_array, 
    y=cap_real, 
    mode='lines', 
    name=f"Case 2 (Real, IBO = {IBO_dB} dB)", 
    line=dict(color='#003366', width=3),
    hovertemplate="Real Capacity: %{y:.2f} Gbps<extra></extra>"
))

fig4.update_layout(
    xaxis_title="Antenna Isolation (dB)", 
    yaxis_title="Channel Capacity (Gbps)", 
    height=500, 
    template="plotly_white",
    hovermode="x unified"
)

# Linea indicatore per l'isolamento scelto sulla sidebar
fig4.add_vline(x=iso_dB, line=dict(color="gray", dash="dot"), annotation_text="Current Isolation", annotation_position="top left")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==========================================
# SEZIONE 5: La Visione Commerciale (Heatmap IBO-Isolamento)
# ==========================================
st.header("5. Strategic Co-Design Map: RF Isolation vs. Baseband IBO")
st.markdown("This Heatmap unites the expertise of two engineering domains within Nokia. To maximize link capacity (the brightest/yellow zones), the system requires simultaneous optimization of the **Hardware RF Isolation** (Y-axis) and the **Baseband Amplifier Back-Off** (X-axis). It visually answers how much isolation is needed to support a given transmit power without succumbing to non-linear distortion.")

Ibo_mesh, Iso_mesh = np.meshgrid(np.linspace(0, 15, 30), np.linspace(40, 80, 30))

# Fast interpolation for distortion variance to ensure smooth UI interaction
dist_vars_interp = np.array([calc_dist_var(i)[0] for i in np.linspace(0, 15, 30)])
P_dist_norm_mesh = np.interp(Ibo_mesh, np.linspace(0, 15, 30), dist_vars_interp)

# Matrix calculations for Link Budget and Capacity
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
    colorbar=dict(title="Capacity (Gbps)"),
    hovertemplate="<b>IBO:</b> %{x:.1f} dB<br><b>Isolation:</b> %{y:.1f} dB<br><b>Capacity:</b> %{z:.2f} Gbps<extra></extra>"
))

fig5.update_layout(
    xaxis_title="Amplifier Input Back-Off (IBO) [dB]", 
    yaxis_title="Antenna Isolation (dB)", 
    height=600, 
    template="plotly_white"
)

# Marker identifying the current operating point
fig5.add_trace(go.Scatter(
    x=[IBO_dB], 
    y=[iso_dB], 
    mode='markers+text', 
    text=["Operating Point"], 
    marker=dict(size=12, color='white', symbol='diamond', line=dict(color='black', width=1.5)), 
    textposition="top center", 
    showlegend=False,
    hoverinfo="skip"
))

st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.link_button( "Dashboard 1",  "https://nokia-band-analysis.streamlit.app/")
st.link_button("Dashboard 3","https://dash-board-2.streamlit.app/")