import numpy as np
import matplotlib.pyplot as plt
import os  # for saving output files

# 0) DIRECTORIES
# Base results directory on Desktop
desktop = os.path.expanduser("~/Desktop")
output_dir = os.path.join(desktop, "neuristor_results")
os.makedirs(output_dir, exist_ok=True)
# Analysis subdirectories
analysis_dir = os.path.join(output_dir, "neuristor_analysis")
meanvar_dir = os.path.join(analysis_dir, "mean_variance")
isi_dir = os.path.join(analysis_dir, "isi_histograms")
amplitude_dir = os.path.join(analysis_dir, "amplitude_distributions")
os.makedirs(meanvar_dir, exist_ok=True)
os.makedirs(isi_dir, exist_ok=True)
os.makedirs(amplitude_dir, exist_ok=True)

# 1) PARAMETERS
C      = 145e-12  # F
Rload  = 12e3     # Ω
Cth    = 49.6e-12 # J/K
Se     = 0.201e-3 # W/K
T0     = 325.0    # K
R0     = 8e-3     # Ω
Ea     = 5220.0   # K
Rm     = 1286.0   # Ω
w      = 7.19     # K
Tc     = 332.8    # K

# 2) HEATER
heater_period = 1e-7
duty_cycle    = 0.5

# 3) HYSTERESIS MODEL
beta = 0.253; eta = 0.956; EPS = 1e-3

def P(x):
    return 0.5 * (1 - np.sin(eta*x)) / (1 + np.tanh(np.pi**2 - 2*np.pi*x))

def F(T, Tr, delta):
    F_tr = 0.5 + 0.5*np.tanh(beta * (delta*(w/2) + Tc - Tr))
    x_at = np.clip(2*F_tr - 1, -0.999999, 0.999999)
    Tpr  = delta*(w/2) + Tc - (1.0/beta) * np.arctanh(x_at) - Tr
    if np.abs(Tpr) < 1e-9:
        Tpr = np.sign(Tpr) * 1e-9 if Tpr != 0 else 1e-9
    xi = (T - Tr) / Tpr
    bracket = T + Tpr * P(xi)
    arg = beta * (delta*(w/2) + Tc - bracket)
    return 0.5 + 0.5*np.tanh(arg)

def R_of_T(T, Tr, delta):
    return R0*np.exp(Ea/T)*F(T,Tr,delta) + Rm

# 4) SIMULATION FUNCTION
def simulate_neuristor(Vin, P_heater, dt, t_total):
    N = int(t_total/dt)
    time = np.arange(N)*dt
    V = np.zeros(N); T = np.zeros(N); I = np.zeros(N)
    V[0]=0.0; T[0]=T0; Tr, delta = T0, +1
    for k in range(N-1):
        Rv = R_of_T(T[k],Tr,delta)
        V[k+1] = V[k] + ((Vin/Rload - V[k]*(1.0/Rv+1.0/Rload))/C)*dt
        P_h = P_heater if (time[k] % heater_period) < heater_period*duty_cycle else 0.0
        T[k+1] = T[k] + ((V[k]**2/Rv + P_h - Se*(T[k]-T0))/Cth)*dt
        if delta>0 and T[k+1]<Tr-EPS: delta,Tr = -1, T[k+1]
        elif delta<0 and T[k+1]>Tr+EPS: delta,Tr = +1, T[k+1]
        I[k+1] = V[k+1]/Rv
    return time, V, I, T

# 5) ANALYSIS SETUP
steady_start = 125e-6  # start time for analysis
steady_end   = 240e-6  # end time for analysis

dt = 5e-8; t_total = 240e-6
heater_powers = [i*1e-3 for i in range(1,11)]
vins = list(np.arange(5,16,5))

# Pre-allocate statistics
mean_T = np.zeros((len(heater_powers), len(vins)))
var_T  = np.zeros_like(mean_T)
isi_data = {}
amp_data = {}

# Run simulations and collect stats within the steady window
for i, P_h in enumerate(heater_powers):
    for j, Vin in enumerate(vins):
        time, V, I, T = simulate_neuristor(Vin, P_h, dt, t_total)
        # Identify indices for steady window
        idx_start = np.searchsorted(time, steady_start)
        idx_end   = np.searchsorted(time, steady_end)
        # Extract steady-state slices
        Tss = T[idx_start:idx_end]
        mean_T[i,j] = Tss.mean()
        var_T[i,j]  = Tss.var()
        # Spike detection and ISI in window
        thr = 0.5 * I.max()
        crossings = np.where((I[1:]>=thr) & (I[:-1]<thr))[0]
        # Keep spikes within window
        spk_idx = crossings[(crossings >= idx_start) & (crossings < idx_end)]
        isi_data[(i,j)] = np.diff(time[spk_idx])
        # Peak amplitudes in window
        Vp, Ip = [], []
        for k in range(len(spk_idx)-1):
            seg = slice(spk_idx[k], spk_idx[k+1])
            Vp.append(V[seg].max())
            Ip.append(I[seg].max())
        amp_data[(i,j)] = (np.array(Vp), np.array(Ip))

# 6) PLOT & SAVE
# Mean & variance vs heater power
for j, Vin in enumerate(vins):
    plt.figure()
    plt.errorbar([p*1e3 for p in heater_powers], mean_T[:,j], yerr=np.sqrt(var_T[:,j]), marker='o')
    plt.xlabel('Heater Power (mW)')
    plt.ylabel('Mean Temp (K)')
    plt.title(f'Mean & STD vs Power, Vin={Vin}V (125–250µs)')
    out = os.path.join(meanvar_dir, f"meanvar_Vin_{Vin}V.png")
    plt.tight_layout(); plt.savefig(out); plt.close()

# Combined summary per (P_h, Vin)
for i, P_h in enumerate(heater_powers):
    for j, Vin in enumerate(vins):
        intervals = isi_data.get((i,j), [])
        Vp, Ip   = amp_data.get((i,j), (np.array([]), np.array([])))
        if len(intervals)==0 or len(Vp)==0:
            continue
        # Simulate again to get raw traces
        time, V_full, I_full, T_full = simulate_neuristor(Vin, P_h, dt, t_total)
        # Recalculate window indices for this simulation
        start_idx = np.searchsorted(time, steady_start)
        end_idx   = np.searchsorted(time, steady_end)
        # Time-current slice for steady window
        t_slice = time[start_idx:end_idx] * 1e6
        I_slice = I_full[start_idx:end_idx] * 1e3
        # Create summary figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # Current vs Time (window)
        axs[0,0].plot(t_slice, I_slice, lw=0.5)
        # Focus x-axis on steady window
        axs[0,0].set_xlim(steady_start*1e6, steady_end*1e6)
        axs[0,0].set(xlabel='Time (µs)', ylabel='I (mA)', title='Current vs Time (125–250 µs)')
        # ISI Histogram
        axs[0,1].hist(intervals * 1e6, bins=50)
        axs[0,1].set(xlabel='ISI (µs)', ylabel='Count', title='ISI Histogram')
        # Voltage amplitude histogram
        axs[1,0].hist(Vp, bins=50)
        axs[1,0].set(xlabel='V Peak (V)', ylabel='Count', title='Voltage Amp Hist')
        dt = 5e-8; t_total = 60e-6  # limit total simulation time to 60 µs
        axs[1,1].hist(Ip * 1e3, bins=50)
        axs[1,1].set(xlabel='I Peak (mA)', ylabel='Count', title='Current Amp Hist')
        fig.tight_layout()
        out = os.path.join(analysis_dir, f"summary_P{int(P_h*1e3)}mW_Vin_{Vin}V.png")
        fig.savefig(out)
        plt.close(fig)

