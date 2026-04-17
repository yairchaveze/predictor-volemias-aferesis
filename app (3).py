import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from datetime import date
import os

st.set_page_config(
    page_title="Predictor de Volemias · YOCE",
    page_icon="💉",
    layout="centered"
)

st.markdown("""
<style>
  .header-box{background:#0d3b6e;border-radius:12px;padding:14px 20px;
    display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}
  .header-left h2{color:#fff;font-size:16px;margin:0;font-family:Arial}
  .header-left p{color:#b5d4f4;font-size:11px;margin:4px 0 0}
  .header-right{text-align:right}
  .yoce{color:#b5d4f4;font-size:20px;font-weight:bold;letter-spacing:3px}
  .vpill{background:rgba(255,255,255,0.15);border-radius:5px;padding:2px 8px;
    font-size:10px;color:#e6f1fb;display:inline-block;margin-top:4px}
  .vdate{color:#85b7eb;font-size:9px;margin-top:3px}
  .slabel{font-size:11px;font-weight:bold;color:#0d3b6e;text-transform:uppercase;
    letter-spacing:1px;margin:14px 0 6px}
  .itip{font-size:10px;color:#555;background:#e6f1fb;border-left:3px solid #185FA5;
    padding:4px 10px;border-radius:4px;margin-top:4px;margin-bottom:8px}
  .result-box{background:#eaf4fb;border-left:6px solid #0d3b6e;border-radius:8px;
    padding:16px;margin-top:16px}
  .res-tipo{font-size:13px;font-weight:bold;color:#0d3b6e}
  .res-date{font-size:10px;color:#888}
  .res-num{font-size:48px;font-weight:bold;color:#0d3b6e;margin:4px 0}
  .res-row{display:flex;justify-content:space-between;padding:7px 0;
    border-bottom:1px solid #ddd;font-size:13px}
  .res-label{color:#555}
  .res-val{font-weight:bold;color:#0d3b6e}
  .badge-ok{background:#EAF3DE;color:#27500A;border-radius:7px;
    padding:7px;text-align:center;font-weight:bold;font-size:12px;margin-top:8px}
  .badge-warn{background:#FAEEDA;color:#633806;border-radius:7px;
    padding:7px;text-align:center;font-weight:bold;font-size:12px;margin-top:8px}
  .mi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px}
  .mi-card{background:#fff;border-radius:6px;padding:6px;text-align:center;border:0.5px solid #ddd}
  .mi-label{font-size:9px;color:#888;margin-bottom:2px}
  .mi-val{font-size:12px;font-weight:bold;color:#333}
  .footer{font-size:9px;color:#aaa;text-align:center;margin-top:10px;
    border-top:0.5px solid #ddd;padding-top:8px;line-height:1.7}
  .calc2-box{background:#f0f7ff;border-left:6px solid #185FA5;border-radius:8px;
    padding:14px;margin-top:14px}
  .calc2-title{font-size:13px;font-weight:bold;color:#185FA5;margin-bottom:4px}
  .calc2-sub{font-size:10px;color:#666;margin-bottom:4px}
  .calc2-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
  .calc2-card{background:#fff;border-radius:6px;padding:8px;text-align:center;border:0.5px solid #b5d4f4}
  .calc2-label{font-size:9px;color:#888;margin-bottom:3px}
  .calc2-val{font-size:15px;font-weight:bold;color:#0d3b6e}
  .vol-note{font-size:10px;color:#666;background:#e8f4fd;border-radius:5px;
    padding:5px 10px;margin-top:6px;text-align:center}
</style>
""", unsafe_allow_html=True)

VER = "v5.0"
MAE = 0.266
CE  = 0.48

st.markdown(f"""
<div class="header-box">
  <div class="header-left">
    <h2>Predictor de volemias · Aféresis CPH</h2>
    <p>CD34+ 10–150 /µL · Adulto ≥18 a · Peso receptor 40–123 kg · Optia · IDL</p>
  </div>
  <div class="header-right">
    <div class="yoce">YOCE</div>
    <div class="vpill">{VER} · GradientBoosting · n=140 · MAE={MAE} · R²=0.852</div>
    <div class="vdate">Última actualización: abril 2026</div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "datos_modelo.csv"))
    df_f = df[
        (df["PRE_CD34"] >= 10) & (df["PRE_CD34"] <= 150) &
        (df["EDAD_REC"] >= 18) & (df["PESO_REC"] >= 40)
    ].copy()
    df_f["VEL_FINAL_IMP"] = df_f["VEL_FINAL"].copy()
    df_f.loc[df_f["ACCESO_BIN"]==1,"VEL_FINAL_IMP"] = df_f.loc[df_f["ACCESO_BIN"]==1,"VEL_FINAL_IMP"].fillna(85.0)
    df_f.loc[df_f["ACCESO_BIN"]==0,"VEL_FINAL_IMP"] = df_f.loc[df_f["ACCESO_BIN"]==0,"VEL_FINAL_IMP"].fillna(71.5)
    FEATURES = ["VOLEMIA","VOL_COSECHA","ACCESO_BIN","VEL_INICIAL","PRE_HTO",
                "VEL_FINAL_IMP","PESO_REC","PESO_DON","PRE_CD34","PRE_WBC",
                "EDAD_DON","PRE_PLT","TIPO_BIN","PRE_MNC_PCT","PRE_MNC_ABS",
                "PLERIXAFOR","DOSIS_GCSF","DIAS_GCSF"]
    data = df_f.dropna(subset=FEATURES + ["TARGET"])
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=3, min_samples_leaf=5,
                                       subsample=0.8, random_state=42)
    model.fit(data[FEATURES], data["TARGET"])
    return model, FEATURES

model, FEATURES = load_and_train()
st.success(f"✅ Modelo {VER} listo — n=140 | MAE={MAE} | R²=0.852")

st.markdown('<div class="slabel">Tipo de acceso y donador</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    acceso = st.radio("Tipo de acceso vascular", ["Catéter","Punción venosa"], horizontal=True)
with col2:
    tipo = st.radio("Tipo de donador", ["Alogénico","Haploidéntico"], horizontal=True)

vel_fin_imp = 85.0 if acceso == "Catéter" else 71.5
st.markdown(f'<div class="itip">Vel. final estimada: <b>{vel_fin_imp} mL/min</b> (mediana institucional · {acceso.lower()})</div>', unsafe_allow_html=True)

st.markdown('<div class="slabel">CD34+ día 5 — variable clave</div>', unsafe_allow_html=True)
cd34 = st.number_input("CD34+ /µL [10–150]", 10.0, 150.0, 50.0, 0.5, label_visibility="collapsed")

st.markdown('<div class="slabel">Donador</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: edad_don = st.number_input("Edad donador (años)", 18, 70, 35, 1)
with c2: peso_don = st.number_input("Peso donador (kg)",   40.0, 140.0, 70.0, 0.5)
with c3: volemia  = st.number_input("Vol. sanguíneo (mL)", 2000, 8000, 4500, 50)

vol_cos_est = round(volemia * 0.052)
st.markdown(f'<div class="vol-note">Vol. cosecha estimado automáticamente: <b>{vol_cos_est} mL</b> (5.2% de volemia · rango institucional: 133–404 mL · mediana: 249 mL)</div>', unsafe_allow_html=True)

st.markdown('<div class="slabel">Receptor y procedimiento</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: peso_rec = st.number_input("Peso receptor (kg) [40–123]", 40.0, 123.0, 70.0, 0.5)
with c2: vel_ini  = st.number_input("Vel. inicial (mL/min)", 30, 120, 65, 1)

st.markdown('<div class="slabel">Biometría pre-aféresis</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: wbc     = st.number_input("WBC pre (k/µL)",     1.0, 100.0, 30.0, 0.1)
with c2: hto     = st.number_input("HTO pre (%)",        20.0, 55.0, 42.0, 0.5)
with c3: plt_val = st.number_input("PLT pre (k/µL)",     50.0, 600.0, 220.0, 5.0)
c1, c2 = st.columns(2)
with c1: mnc_abs = st.number_input("MNC absoluto (k/µL)", 0.0, 25.0, 3.0, 0.1)
with c2: mnc_pct = st.number_input("MNC% pre",            0.0, 60.0, 10.0, 0.5)

st.markdown('<div class="slabel">G-CSF y Plerixafor</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: dias_gcsf  = st.number_input("Días G-CSF", 1, 8, 4, 1)
with c2: dosis_gcsf = st.number_input("Dosis G-CSF (µg/día)", 300, 1200, 600, 50)

plerix_on = st.checkbox("Plerixafor administrado")
st.caption("Al activar, ingresa la dosis real. Rango institucional: 0.12–0.27 mg/kg · Mediana: 0.24 mg/kg")

plerix_val   = 0.0
plerix_dosis = 0.24
if plerix_on:
    plerix_dosis = st.number_input("Dosis plerixafor (mg/kg)", 0.10, 0.30, 0.24, 0.01)
    plerix_val   = 1.0
    total_mg     = round(peso_don * plerix_dosis, 1)
    st.markdown(f"**{total_mg} mg totales** ({peso_don} kg × {plerix_dosis} mg/kg)")

st.markdown("<br>", unsafe_allow_html=True)
calcular = st.button("Calcular volemias recomendadas", type="primary", use_container_width=True)

if calcular:
    X_new = pd.DataFrame([{
        "VOLEMIA"      : volemia,
        "VOL_COSECHA"  : vol_cos_est,
        "ACCESO_BIN"   : float(1 if acceso=="Catéter" else 0),
        "VEL_INICIAL"  : vel_ini,
        "PRE_HTO"      : hto,
        "VEL_FINAL_IMP": vel_fin_imp,
        "PESO_REC"     : peso_rec,
        "PESO_DON"     : peso_don,
        "PRE_CD34"     : cd34,
        "PRE_WBC"      : wbc,
        "EDAD_DON"     : edad_don,
        "PRE_PLT"      : plt_val,
        "TIPO_BIN"     : 1 if tipo=="Haploidéntico" else 0,
        "PRE_MNC_PCT"  : mnc_pct,
        "PRE_MNC_ABS"  : mnc_abs,
        "PLERIXAFOR"   : plerix_val,
        "DOSIS_GCSF"   : dosis_gcsf,
        "DIAS_GCSF"    : dias_gcsf,
    }])

    vol_pred = model.predict(X_new)[0]
    vol_low  = max(1.0, vol_pred - MAE)
    vol_high = vol_pred + MAE
    cd34_est = (cd34 * volemia * vol_pred * CE) / (peso_rec * 1000)
    ok       = 5 <= cd34_est <= 10
    hoy      = date.today().strftime("%d/%m/%Y")
    plx_str  = f" · Plerixafor {plerix_dosis} mg/kg" if plerix_on else ""
    badge_class = "badge-ok" if ok else "badge-warn"
    badge_txt   = "En rango objetivo 5–10 ×10⁶/kg" if ok else "Fuera del rango objetivo"

    st.markdown(f"""
    <div class="result-box">
      <div style="display:flex;justify-content:space-between;margin-bottom:10px">
        <div>
          <div class="res-tipo">{tipo} · {acceso}{plx_str}</div>
          <div class="res-date">HU-UANL · {hoy}</div>
        </div>
        <div style="text-align:right;font-size:10px;color:#aaa">YOCE<br>{VER}</div>
      </div>
      <div class="res-num">{vol_pred:.2f}</div>
      <div style="color:#666;font-size:14px">volemias recomendadas</div>
      <div class="res-row">
        <span class="res-label">Rango de confianza (±MAE)</span>
        <span class="res-val" style="color:#1a6fa8">{vol_low:.2f} – {vol_high:.2f} vol</span>
      </div>
      <div class="res-row">
        <span class="res-label">CD34+ estimado en receptor</span>
        <span class="res-val">{cd34_est:.1f} ×10⁶/kg</span>
      </div>
      <div class="res-row">
        <span class="res-label">Vol. cosecha utilizado por el modelo</span>
        <span class="res-val">{vol_cos_est} mL (5.2% de volemia)</span>
      </div>
      <div class="{badge_class}">{badge_txt}</div>
      <div class="mi-grid">
        <div class="mi-card"><div class="mi-label">Versión</div><div class="mi-val">{VER}</div></div>
        <div class="mi-card"><div class="mi-label">n</div><div class="mi-val">140</div></div>
        <div class="mi-card"><div class="mi-label">MAE</div><div class="mi-val">{MAE} vol</div></div>
        <div class="mi-card"><div class="mi-label">R²</div><div class="mi-val">0.852</div></div>
      </div>
      <div class="footer">
        GradientBoostingRegressor · 18 variables · Spectra Optia · Kit IDL · HU-UANL<br>
        CD34+ 10–150 /µL · Vel. final: catéter=85 · punción=71.5 mL/min · CE=48%<br>
        Desarrollado por YOCE — Yair Omar Chávez Estrada · abril 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="calc2-box">
      <div class="calc2-title">Calculador de escenarios</div>
      <div class="calc2-sub">¿Qué pasa si proceso X volemias con Y mL de cosecha? Ajusta de 0.1 en 0.1</div>
    </div>
    """, unsafe_allow_html=True)

    vol_sim = st.number_input("Volemias a procesar",
                               min_value=1.0, max_value=9.0,
                               value=round(vol_pred, 1), step=0.1)
    vol_cos_sim = st.number_input("Vol. cosecha programado (mL)",
                                   min_value=133, max_value=404,
                                   value=vol_cos_est, step=5)

    cd34_sim   = (cd34 * volemia * vol_sim * CE) / (peso_rec * 1000)
    sangre_sim = volemia * vol_sim
    vel_prom   = (vel_ini + vel_fin_imp) / 2
    tiempo_sim = round(sangre_sim / vel_prom)
    ok_sim     = 5 <= cd34_sim <= 10
    badge_sim  = "badge-ok" if ok_sim else "badge-warn"
    badge_stxt = "En rango objetivo 5–10 ×10⁶/kg" if ok_sim else "Fuera del rango objetivo"

    st.markdown(f"""
    <div class="calc2-grid">
      <div class="calc2-card">
        <div class="calc2-label">CD34+ estimado receptor</div>
        <div class="calc2-val">{cd34_sim:.1f} ×10⁶/kg</div>
      </div>
      <div class="calc2-card">
        <div class="calc2-label">Sangre total procesada</div>
        <div class="calc2-val">{sangre_sim:,.0f} mL</div>
      </div>
      <div class="calc2-card">
        <div class="calc2-label">Tiempo estimado</div>
        <div class="calc2-val">{tiempo_sim} min</div>
      </div>
    </div>
    <div class="{badge_sim}" style="margin-top:10px;padding:7px;border-radius:7px;
      font-weight:bold;font-size:12px;text-align:center">{badge_stxt}</div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Historial de versiones"):
    versiones = [
        ("v5.0","Abril 2026","140","0.266","0.852",
         "Rango CD34+ ampliado a 10-150 /µL (+42 casos). Analisis de importancia confirmo necesidad de las 18 variables (reduccion a 13 cambia solo 0.011 vol en media, sin impacto clinico). VOL_COSECHA reemplazada por estimacion automatica 5.2% de volemia. Calculador de escenarios con step de 0.1 volemias."),
        ("v4.5","Abril 2026","98","0.327","0.737",
         "App web Streamlit. Plerixafor con dosis ajustable y mg totales automaticos."),
        ("v4.0","Abril 2026","98","0.327","0.737",
         "Vel. final imputada por mediana institucional segun acceso."),
        ("v3.0","Abril 2026","98","0.362","0.690",
         "Se elimina vel. final del panel de entrada."),
        ("v2.0","Abril 2026","98","0.327","0.737",
         "Velocidades, vol. cosecha y filtro peso receptor >= 40 kg."),
        ("v1.0","Abril 2026","279","0.536","0.642",
         "Modelo inicial ALO+HAPLO sin filtros de poblacion."),
    ]
    for v, fecha, n, mae, r2, desc in versiones:
        txt = "**" + v + "** · " + fecha + " · n=" + n + " · MAE=" + mae + " · R2=" + r2 + "  \n" + desc
        st.markdown(txt)
        st.divider()
