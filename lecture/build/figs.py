import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
INK="#1F2328"; RED="#C8102E"; GRAY="#9AA3AF"; MUTED="#6B7280"; LIGHT="#E5E7EB"
rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"cm","axes.edgecolor":LIGHT,
                 "axes.labelcolor":MUTED,"xtick.color":MUTED,"ytick.color":MUTED})
OUT="img/"
def clean(ax):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

# --- 1. ResNet = Euler steps vs ODE curve
f=lambda h: 0.9*np.sin(1.1*h)+0.35
fig,ax=plt.subplots(figsize=(5.2,3.4),dpi=300)
T=np.linspace(0,4,2000); h=np.zeros_like(T); h[0]=0.2
for i in range(1,len(T)): h[i]=h[i-1]+(T[i]-T[i-1])*f(h[i-1])
ax.plot(T,h,color=RED,lw=3,label="ODE: continuous depth",zorder=3)
ts=np.arange(0,5); hs=[0.2]
for i in range(1,5): hs.append(hs[-1]+f(hs[-1]))
ax.plot(ts,hs,"-o",color=INK,lw=2,ms=7,alpha=.9,label="ResNet: one block = one Euler step",zorder=4)
ax.set_xlabel("depth  /  time",fontsize=12); ax.set_ylabel("hidden state",fontsize=12)
clean(ax); ax.legend(frameon=False,fontsize=11,loc="lower right")
fig.tight_layout(); fig.savefig(OUT+"resnet_vs_ode.png",transparent=True); plt.close()

# --- 2. ODE one path vs SDE many paths (same drift)
rng=np.random.default_rng(3)
mu=lambda x: 1.2*np.sin(x)+0.3
t=np.linspace(0,3,600); dt=t[1]-t[0]
def sim(sig,n):
    X=np.zeros((n,len(t))); X[:,0]=0.5
    for i in range(1,len(t)):
        X[:,i]=X[:,i-1]+mu(X[:,i-1])*dt+sig*np.sqrt(dt)*rng.standard_normal(n)
    return X
for name,sig,n in [("ode_path",0.0,1),("sde_paths",0.55,40)]:
    fig,ax=plt.subplots(figsize=(4.2,3.0),dpi=300)
    X=sim(sig,n)
    if n>1:
        for x in X[1:]: ax.plot(t,x,color=GRAY,lw=.9,alpha=.55)
        ax.plot(t,X[0],color=RED,lw=2.2)
    else: ax.plot(t,X[0],color=RED,lw=3)
    ax.set_ylim(-2.6,5.6); clean(ax)
    ax.set_xlabel("time",fontsize=12)
    fig.tight_layout(); fig.savefig(OUT+name+".png",transparent=True); plt.close()

# --- 3. Diffusion: data -> noise forward paths (VP / OU process), densities at both ends
n=300; T=1.0; N=500; tt=np.linspace(0,T,N); dt=tt[1]-tt[0]
x0=np.where(rng.random(n)<.5, rng.normal(-2,.25,n), rng.normal(2,.25,n))
beta=lambda s: 0.1+ (12-0.1)*s
X=np.zeros((n,N)); X[:,0]=x0
for i in range(1,N):
    b=beta(tt[i-1]); X[:,i]=X[:,i-1]-0.5*b*X[:,i-1]*dt+np.sqrt(b*dt)*rng.standard_normal(n)
fig=plt.figure(figsize=(8.6,3.2),dpi=300)
gs=fig.add_gridspec(1,3,width_ratios=[1,6,1],wspace=0.04)
axL=fig.add_subplot(gs[0]); ax=fig.add_subplot(gs[1],sharey=axL); axR=fig.add_subplot(gs[2],sharey=axL)
for x in X[:60]: ax.plot(tt,x,color=GRAY,lw=.7,alpha=.6)
for x in X[[3,7]]: ax.plot(tt,x,color=RED,lw=1.8)
ys=np.linspace(-3.5,3.5,300)
from math import pi
dens0=0.5*(np.exp(-(ys+2)**2/(2*.25**2))+np.exp(-(ys-2)**2/(2*.25**2)))/(.25*np.sqrt(2*pi))
dens1=np.exp(-ys**2/2)/np.sqrt(2*pi)
axL.fill_betweenx(ys,0,dens0,color=INK,alpha=.85); axL.invert_xaxis()
axR.fill_betweenx(ys,0,dens1,color=MUTED,alpha=.8)
for a in (axL,ax,axR): clean(a); a.set_ylim(-3.5,3.5)
for a in (axL,axR):
    for s in a.spines.values(): s.set_visible(False)
axL.set_xlabel("data",fontsize=12,color=INK); axR.set_xlabel("noise",fontsize=12,color=INK)
ax.set_xlabel("time  →",fontsize=12)
fig.savefig(OUT+"diffusion_paths.png",transparent=True,bbox_inches="tight"); plt.close()

# --- 4. One Brownian path, reused
W=np.cumsum(np.sqrt(1/400)*rng.standard_normal(400)); W-=W[0]
fig,ax=plt.subplots(figsize=(4.6,2.4),dpi=300)
ax.plot(np.linspace(0,1,400),W,color=INK,lw=1.6)
clean(ax); ax.set_xlabel("time",fontsize=12); ax.set_ylabel("W(t)",fontsize=12)
fig.tight_layout(); fig.savefig(OUT+"brownian.png",transparent=True); plt.close()

# --- Equations (mathtext), dark ink and white variants
def eq(name, tex, size=26, color=INK):
    fig=plt.figure(figsize=(0.01,0.01),dpi=300)
    fig.text(0,0,tex,fontsize=size,color=color)
    fig.savefig(OUT+name+".png",transparent=True,bbox_inches="tight",pad_inches=0.04); plt.close()
eq("eq_resnet", r"$h_{k+1} = h_k + f_\theta(h_k)$")
eq("eq_ode",    r"$\dfrac{dh}{dt} = f_\theta(h, t)$")
eq("eq_node",   r"$z(T) = z(0) + \int_0^T f_\theta(z(t),\,t)\,dt$")
eq("eq_adj1",   r"$a(t) = \dfrac{\partial L}{\partial z(t)}$",24)
eq("eq_adj2",   r"$\dfrac{da}{dt} = -\,a(t)^{\top}\dfrac{\partial f_\theta}{\partial z}$",24)
eq("eq_adj3",   r"$\dfrac{dL}{d\theta} = \int_0^T a(t)^{\top}\dfrac{\partial f_\theta}{\partial \theta}\,dt$",24)
eq("eq_sde",    r"$dZ_t = \mu_\theta(Z_t, t)\,dt \;+\; \sigma_\theta(Z_t, t)\,dW_t$",30)
eq("eq_em",     r"$Z_{k+1} = Z_k + \mu_\theta\,\Delta t + \sigma_\theta\,\sqrt{\Delta t}\;\varepsilon_k,\quad \varepsilon_k \sim \mathcal{N}(0, I)$",20)
eq("eq_fwd",    r"$dx = f(x,t)\,dt + g(t)\,dW$",22)
eq("eq_rev",    r"$dx = \left[f(x,t) - g(t)^2\, s_\theta(x,t)\right]dt + g(t)\,d\bar{W}$",22)
eq("eq_pf",     r"$\dfrac{dx}{dt} = f(x,t) - \frac{1}{2}\,g(t)^2\, s_\theta(x,t)$",22)
eq("eq_score",  r"$s_\theta(x,t) \approx \nabla_x \log p_t(x)$",20, MUTED)
print("done")

# --- title-slide paths on dark
rng=np.random.default_rng(11)
X=sim(0.55,28)
fig,ax=plt.subplots(figsize=(4.4,1.9),dpi=300)
for x in X[1:]: ax.plot(t,x,color="#6B7280",lw=.8,alpha=.7)
ax.plot(t,X[0],color="#E8475E",lw=2)
ax.axis("off"); fig.tight_layout(pad=0); fig.savefig(OUT+"sde_paths_dark.png",transparent=True); plt.close()


# =====================================================================
# v2 (grad-level deck): running example + new equations
# =====================================================================
rng = np.random.default_rng(7)

# --- schematic: latent quality fan chart for three storage temperatures, threshold, shelf-life histogram
def quality_paths(k, sig, n=600, T=20.0, N=1000):
    tt = np.linspace(0, T, N); dt = tt[1] - tt[0]
    q = np.zeros((n, N)); q[:, 0] = rng.normal(1.0, 0.15, n)
    for i in range(1, N):
        q[:, i] = q[:, i-1] + k * (0.35 + 0.08 * q[:, i-1]) * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
    return tt, q

B = 4.0
fig = plt.figure(figsize=(8.4, 3.3), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.25], wspace=0.08)
ax = fig.add_subplot(gs[0]); axh = fig.add_subplot(gs[1], sharey=None)
cols = {"10 °C": "#9AA3AF", "20 °C": RED, "ambient": INK}
taus = {}
for (lab, k, sig) in [("10 °C", 0.38, 0.22), ("20 °C", 0.75, 0.30), ("ambient", 0.95, 0.38)]:
    tt, q = quality_paths(k, sig)
    lo, med, hi = np.percentile(q, [10, 50, 90], axis=0)
    ax.fill_between(tt, lo, hi, color=cols[lab], alpha=0.18, lw=0)
    ax.plot(tt, med, color=cols[lab], lw=2.2, label=lab)
    hit = (q >= B)
    tau = np.where(hit.any(1), tt[np.argmax(hit, 1)], np.nan)
    taus[lab] = tau[~np.isnan(tau)]
ax.axhline(B, color=INK, ls="--", lw=1.2)
ax.text(19.8, B + 0.12, "end of shelf life (stage 4)", fontsize=10, color=INK, ha="right")
ax.set_xlim(0, 20); ax.set_ylim(0.3, 5.6)
ax.set_xlabel("days in storage", fontsize=11); ax.set_ylabel("latent ripeness  q(Z_t)", fontsize=11)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=9)
ax.legend(frameon=False, fontsize=9, loc="lower right", title="storage", title_fontsize=9)
for lab in cols:
    axh.hist(taus[lab], bins=np.arange(0, 20.5, 0.5), color=cols[lab], histtype="step", lw=2)
axh.set_xlabel("shelf life τ (days)", fontsize=10); axh.set_yticks([])
for s in ["top", "right", "left"]: axh.spines[s].set_visible(False)
axh.tick_params(labelsize=9)
axh.set_title("schematic · synthetic data", fontsize=8, color=MUTED, style="italic", loc="right")
fig.savefig(OUT + "ripening_fan.png", transparent=True, bbox_inches="tight"); plt.close()

# --- indicator vs. smooth surrogate: E[1{Z_T > b}] as a function of a drift parameter
th = np.linspace(-0.6, 1.4, 200)
W = rng.standard_normal(8)
fig, ax = plt.subplots(figsize=(5.6, 2.3), dpi=300)
for w in W[:6]:
    ax.plot(th, (th + 0.6 * w > 0.5).astype(float), color=GRAY, lw=1.0, alpha=0.8, drawstyle="steps-post")
from math import erf
Ef = 0.5 * (1 + np.vectorize(erf)((th - 0.5) / (0.6 * np.sqrt(2))))
ax.plot(th, Ef, color=RED, lw=3, label="E[ 1{spoiled} ]  (smooth in θ)")
ax.plot([], [], color=GRAY, lw=1, label="single paths: step, gradient = 0")
ax.set_xlabel("drift parameter θ", fontsize=12); ax.set_ylabel("spoiled by t*", fontsize=12)
ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=9)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.0, 1.02))
fig.tight_layout(); fig.savefig(OUT + "indicator.png", transparent=True); plt.close()

# --- equations (v2)
E = dict(size=22)
eq("v2_node_food",  r"$\dfrac{dz}{dt} = f_\theta(z, t, c),\qquad z(0) = e_\phi(x_0)$", 22)
eq("v2_aug",        r"$\dfrac{d}{dt}\begin{bmatrix} z \\ a \\ a_\theta \end{bmatrix}$", 22) if False else None
eq("v2_aug1",       r"$\dot z = f_\theta(z,t)$", 20)
eq("v2_aug2",       r"$\dot a = -\,a^{\top}\,\partial_z f_\theta$", 20)
eq("v2_aug3",       r"$\dot a_\theta = -\,a^{\top}\,\partial_\theta f_\theta,\quad a_\theta(T) = 0 \;\Rightarrow\; \dfrac{dL}{d\theta} = a_\theta(0)$", 20)
eq("v2_ito_def",    r"$Z_t = Z_0 + \int_0^t \mu(Z_s,s)\,ds + \int_0^t \sigma(Z_s,s)\,dW_s$", 21)
eq("v2_bm",         r"$W_t - W_s \sim \mathcal{N}(0,\,(t-s)I),\qquad (dW)^2 = dt$", 20)
eq("v2_itolemma",   r"$df(Z_t) = f'(Z_t)\,dZ_t + \frac{1}{2} f''(Z_t)\,\sigma^2\,dt$", 20)
eq("v2_conv",       r"$\mu_{\mathrm{Strat}} = \mu_{\mathrm{It\^o}} - \frac{1}{2}\sum_j (\partial_z \sigma_j)\,\sigma_j$", 20)
eq("v2_ripen",      r"$dZ_t = \kappa(c)\,\mu_\theta(Z_t,t)\,dt + \sigma_\theta(Z_t,t,c)\,dW_t$", 24)
eq("v2_arrh",       r"$\kappa(c) = \exp\!\left[-\frac{E_a}{R}\left(\frac{1}{T_c} - \frac{1}{T_{\mathrm{ref}}}\right)\right]$", 18)
eq("v2_tau",        r"$\tau = \inf\{t : q(Z_t) \geq b\},\qquad \mathbb{P}(\tau \leq t^\ast \mid x_{0:k}, c)$", 20)
eq("v2_query",      r"$\mathbb{P}(\tau \leq t^\ast \mid x_{0:k},\, c)$", 26)
eq("v2_pathwise",   r"$\nabla_\theta\,\mathbb{E}[f(Z_T)] = \mathbb{E}\left[\nabla_\theta f(Z_T(\theta, W))\right]$", 19)
eq("v2_score",      r"$\nabla_\theta\,\mathbb{E}[f(Z)] = \mathbb{E}\left[f(Z)\,\nabla_\theta \log p_\theta(Z)\right]$", 19)
eq("v2_bridge",     r"$\mathbb{P}(\mathrm{cross}\ b \mid q_k, q_{k+1}) = \exp\!\left(-\frac{2(b-q_k)(b-q_{k+1})}{\sigma^2 \Delta t}\right)$", 19)
eq("v2_sadj",       r"$da_t = -\,a_t^{\top}\partial_z\mu\,dt \;-\; \sum_j a_t^{\top}\partial_z\sigma_j \circ dW_t^{j}$", 20)
eq("v2_sadj_th",    r"$\frac{dL}{d\theta} = \int_0^T a_t^{\top}\partial_\theta\mu\,dt + \sum_j \int_0^T a_t^{\top}\partial_\theta\sigma_j \circ dW_t^{j}$", 20)
eq("v2_vbt",        r"$W_s \mid W_u, W_v \;\sim\; \mathcal{N}\!\left(W_u + \frac{s-u}{v-u}(W_v - W_u),\ \frac{(s-u)(v-s)}{v-u}\right)$", 19)
eq("v2_prior",      r"$\mathrm{prior}\quad dZ_t = h_\theta(Z_t,t,c)\,dt + \sigma(Z_t,t,c)\,dW_t$", 19)
eq("v2_post",       r"$\mathrm{posterior}\quad dZ_t = h_\phi(Z_t,t,c,x_{0:K})\,dt + \sigma(Z_t,t,c)\,dW_t$", 19)
eq("v2_elbo",       r"$\log p(y_{1:K}) \geq \mathbb{E}_Q\!\left[\sum_k \log p(y_k \mid Z_{t_k})\right] - \mathrm{KL}_0 - \mathbb{E}_Q\!\int_0^T \frac{1}{2}\left\|u_t\right\|^2 dt$", 19)
eq("v2_u",          r"$\sigma\,u_t = h_\phi - h_\theta$", 19)
eq("v2_dsm",        r"$\min_\theta\ \mathbb{E}_{t,x_0,x_t}\left\| s_\theta(x_t,t) - \nabla_{x_t}\log p_t(x_t \mid x_0)\right\|^2$", 19)
eq("v2_lag",        r"$\mathcal{L} = L(z(T)) - \int_0^T a(t)^{\top}\left(\dot z - f_\theta(z,t)\right)dt$", 20)
eq("v2_lag2",       r"$\frac{d\mathcal{L}}{d\theta} = \int_0^T a^{\top}\partial_\theta f\,dt\quad\mathrm{when}\quad \dot a = -a^{\top}\partial_z f,\ \ a(T) = \partial_z L$", 20)
eq("v2_girs1",      r"$\frac{dQ}{dP} = \exp\!\left(\int_0^T u_t^{\top} dW_t^{P} - \frac{1}{2}\int_0^T \left\|u_t\right\|^2 dt\right)$", 20)
eq("v2_girs2",      r"$\mathrm{KL}(Q\,\|\,P) = \mathbb{E}_Q\!\left[\log\frac{dQ}{dP}\right] = \mathbb{E}_Q\!\int_0^T \frac{1}{2}\left\|u_t\right\|^2 dt$", 20)
eq("v2_strat_ode",  r"$dZ_t = \mu\,dt + \sigma \circ dW_t\quad\Longleftrightarrow\quad \dot Z = \mu + \sigma\,\dot W\ \ (\mathrm{ordinary\ chain\ rule})$", 19)
print("v2 done")


# =====================================================================
# v3: charts derived from Xavier et al. (2024) Tables 4 and 6
# =====================================================================
import math
R_GAS = 8.314
ALPHA = {"T10": 4.390, "T20": 2.116, "Tamb": 1.929}       # days per stage (Table 4, 5-stage)
TEMP = {"T10": 10.0, "T20": 20.0, "Tamb": 18.7}            # °C (Tamb = reported mean)
rate = {k: 1 / a for k, a in ALPHA.items()}
T1, T2 = 283.15, 293.15
Ea = R_GAS * math.log(rate["T20"] / rate["T10"]) / (1 / T1 - 1 / T2)

# Arrhenius plot
fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)
xs = np.linspace(1 / 296.0, 1 / 281.0, 50)
lnA = math.log(rate["T20"]) + Ea / (R_GAS * T2)
ax.plot(xs * 1000, lnA - Ea / R_GAS * xs, color=GRAY, lw=1.5, ls="--", label=f"Arrhenius fit, $E_a$ = {Ea/1000:.0f} kJ/mol")
for k in ("T10", "T20"):
    ax.plot(1000 / (TEMP[k] + 273.15), math.log(rate[k]), "o", color=INK, ms=7)
    ax.annotate(k, (1000 / (TEMP[k] + 273.15), math.log(rate[k])), textcoords="offset points", xytext=(-30 if k == "T20" else 8, 6 if k == "T20" else -4), fontsize=10)
Ta = TEMP["Tamb"] + 273.15
pred = lnA - Ea / (R_GAS * Ta)
ax.plot(1000 / Ta, pred, "o", mfc="white", mec=INK, ms=7, label="Tamb, predicted (18.7 °C)")
ax.plot(1000 / Ta, math.log(rate["Tamb"]), "o", color=RED, ms=8, label="Tamb, observed")
ax.annotate("", xy=(1000 / Ta, math.log(rate["Tamb"])), xytext=(1000 / Ta, pred), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
ax.set_xlabel("1000 / T  (1/K)", fontsize=11); ax.set_ylabel("ln k  (stages / day)", fontsize=11)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=9); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "v3_arrhenius.png", transparent=True); plt.close()

# first-passage ratio check
MAE = {"T10": 1.70, "T20": 0.73, "Tamb": 0.53}          # Table 6, attributed classifications, 5-stage
fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=300)
groups = ["T10", "Tamb"]
for i, g in enumerate(groups):
    r = rate["T20"] / rate[g]
    ax.plot([i], [r ** 1.5], marker="s", color=GRAY, ms=9, ls="none", label=r"constant $\sigma$:  $(v_{20}/v)^{3/2}$" if i == 0 else None)
    ax.plot([i], [r], marker="D", color=INK, mfc="white", ms=8, ls="none", label=r"$\sigma^2 \propto v$, or quantization only:  $v_{20}/v$" if i == 0 else None)
    ax.plot([i], [MAE[g] / MAE["T20"]], marker="o", color=RED, ms=10, ls="none", label="observed MAE ratio" if i == 0 else None)
ax.axhline(1, color=GRAY, lw=0.8, ls=":")
ax.set_xticks([0, 1]); ax.set_xticklabels(["T10 / T20", "Tamb / T20"], fontsize=12)
ax.set_xlim(-0.6, 1.6); ax.set_ylabel("error ratio vs. T20", fontsize=11)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=11); ax.legend(frameon=False, fontsize=9.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "v3_fpt_ratio.png", transparent=True); plt.close()

eq("v3_drift",  r"$\mathrm{DaysLeft} = \alpha_g\,(s - 5)\quad\Rightarrow\quad \dfrac{ds}{dt} \approx v_g = \dfrac{1}{|\alpha_g|}$", 20)
eq("v3_ig",     r"$dX = v\,dt + \sigma\,dW,\ \ \tau = \inf\{t: X_t = L\}\ \ \Rightarrow\ \ \mathbb{E}\tau = \frac{L}{v},\ \ \mathrm{sd}(\tau) = \sqrt{\frac{L\sigma^2}{v^3}}$", 19)
eq("v3_ord",    r"$p(y \leq k \mid Z_t) = \mathrm{sigmoid}\!\left(c_k - q(Z_t)\right),\quad k = 1,\dots,4$", 19)
eq("v3_arrh_fit", r"$E_a = \frac{R\,\ln(v_{20}/v_{10})}{1/T_{10} - 1/T_{20}}$", 20)
eq("v3_ig_den", r"$f_\tau(t) = \frac{L}{\sigma\sqrt{2\pi t^3}}\exp\!\left(-\frac{(L - vt)^2}{2\sigma^2 t}\right)$", 20)
eq("v3_ratio",  r"$\frac{\mathrm{sd}(\tau_{10})}{\mathrm{sd}(\tau_{20})} = \left(\frac{v_{20}}{v_{10}}\right)^{3/2}\ (\sigma\ \mathrm{fixed}),\qquad \frac{v_{20}}{v_{10}}\ \ (\sigma^2 \propto v)$", 20)
print("v3 done", round(Ea/1000, 1))
eq("v3_ig_a", r"$dX = v\,dt + \sigma\,dW,\qquad \tau = \inf\{t: X_t = L\}$", 20)
eq("v3_ig_b", r"$\mathbb{E}\,\tau = \frac{L}{v},\qquad \mathrm{sd}(\tau) = \sqrt{L\sigma^2 / v^3}$", 20)

# =====================================================================
# v4: title-slide graphic (sample paths of a drifted SDE, no axes)
# =====================================================================
rng = np.random.default_rng(7)
n, steps = 40, 400
t = np.linspace(0, 1, steps + 1)
dW = rng.normal(0, np.sqrt(1 / steps), (n, steps))
X = np.zeros((n, steps + 1))
for k in range(steps):
    X[:, k + 1] = X[:, k] + 1.2 / steps + 0.45 * dW[:, k]
fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
mid = int(np.argmin(np.abs(X - X.mean(0)).mean(1)))   # path closest to the mean
for i in range(n):
    if i != mid:
        ax.plot(t, X[i], color="#C9CED6", lw=0.9, alpha=0.9)
ax.plot(t, X[mid], color=RED, lw=2.2)
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT + "title_paths.png", transparent=True); plt.close()
print("v4 done")
