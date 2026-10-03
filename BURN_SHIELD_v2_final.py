
# =====================================================================
# BURN-SHIELD v3
#  - NEW  Login / Create-account first page, PBKDF2-hashed passwords,
#         account lockout, 30-min session timeout, logout.
#  - NEW  More accurate detection: lot-relative robust signals for baseline,
#         early/mid/late drift (per 24h), acceleration and a cross-validated
#         168h surprise + Isolation Forest. Risk is judged by strength of
#         evidence (not "always flag the top 25%").
#  - NEW  Choose Critical / Watch / Normal to view, explain and investigate.
#  - NEW  Sensitivity setting (Sensitive / Balanced / Strict) on Analyze page.
#  - KEEP step-by-step explanations, flagged components explained one by one,
#         upload security (validation, injection protection, rate limit).
#
# Optional environment variables:
#   BURN_SHIELD_INVITE  - if set, this invite code is required to create an account
#   BURN_SHIELD_USERS   - path of the user file (default: burn_shield_users.json next to this file)
# Add burn_shield_users.json to .gitignore (it holds password hashes).
# Recommended .streamlit/config.toml:
#   [server]
#   maxUploadSize = 10
#   enableXsrfProtection = true
#   enableCORS = true
#   [browser]
#   gatherUsageStats = false
# =====================================================================
import io
import os
import re
import json
import html
import hmac
import time
import hashlib
import logging
import threading
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict

st.set_page_config(page_title="BURN-SHIELD", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
log = logging.getLogger("burn_shield")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:"DM Sans",sans-serif}
.stApp{background:#061525;color:#eaf4ff}
.block-container{max-width:1400px;padding-top:1.5rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#071d2e;border-right:1px solid #123f52}
section[data-testid="stSidebar"] *{color:#dff3fb}
.hero{background:linear-gradient(135deg,#0a263d,#0a3b4b);border:1px solid #18566c;border-radius:28px;padding:2.7rem;box-shadow:0 18px 55px #0005}
.hero.center{text-align:center}
.hero h1{font:800 3.2rem Manrope,sans-serif;margin:.35rem 0;color:#f5fcff;letter-spacing:-.04em}
.hero h2{font:800 2rem Manrope,sans-serif;margin:.35rem 0;color:#f5fcff}
.hero p{color:#b9d5e2;font-size:1.02rem;line-height:1.6;max-width:850px}
.badge{display:inline-block;padding:.35rem .7rem;border-radius:999px;background:#123d52;color:#bfeaf0;font-weight:700;font-size:.76rem;letter-spacing:.04em}
.card{background:#0a293d;border:1px solid #15566d;border-radius:20px;padding:1.25rem 1.35rem;margin-bottom:1rem}
.card h3{font:700 1.15rem Manrope,sans-serif;color:#f2fbff;margin:.1rem 0 .45rem}
.muted{color:#9fc0d0;line-height:1.55}
.section{font:800 1.45rem Manrope,sans-serif;color:#f2fbff;margin:1.6rem 0 .45rem}
.metric{background:#092438;border:1px solid #14556c;border-radius:18px;padding:1rem}
.metric .label{font-size:.72rem;color:#8fb5c7;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.metric .value{font:800 1.9rem Manrope,sans-serif;color:#f3fbff;margin-top:.2rem}
.flow{display:flex;gap:.65rem;align-items:stretch;margin:1rem 0 1.4rem}
.flowbox{flex:1;background:#0a293d;border:1px solid #15566d;border-radius:16px;padding:1rem;text-align:center}
.flowbox b{display:block;color:#f1fbff}.flowbox span{font-size:.78rem;color:#8fb5c7}
.step{background:#0a293d;border:1px solid #15566d;border-radius:15px;padding:1rem 1.15rem;margin:.55rem 0}
.step.active{border-color:#39b7c6;box-shadow:0 0 0 1px #39b7c6 inset}
.step.done{border-color:#277f86}
.stButton button{border-radius:12px;font-weight:700;background:#0e6172;color:white;border:1px solid #23879a}
.stButton button:hover{background:#117285;color:white}
div[data-testid="stFileUploader"]{background:#082338;border:1px dashed #2c7e91;border-radius:18px;padding:1rem}
div[data-testid="stDataFrame"]{border-radius:15px;overflow:hidden}
</style>
""", unsafe_allow_html=True)

TIMEPOINTS=[0,24,96,168]
VALUE_COLS=["Value_0h","Value_24h","Value_96h","Value_168h"]
REQUIRED=["Component_ID",*VALUE_COLS]
OPTIONAL=["Lot_ID","Anomaly_Label"]

# ---------- SECURITY SETTINGS ----------
MAX_UPLOAD_MB=10          # maximum CSV size
MAX_ROWS=50_000           # maximum components per file
MAX_COLS=60               # maximum columns in the file
MIN_ROWS=12               # minimum components needed for a meaningful analysis
MAX_ABS_VALUE=1e9         # reject absurd measurement values
MAX_RUNS_PER_MIN=6        # analysis rate limit per session
MAX_LOGIN_TRIES=5         # wrong passwords before an account is locked
LOCKOUT_SEC=300           # account lock time (5 minutes)
SESSION_TIMEOUT=30*60     # auto logout after 30 minutes of inactivity
PBKDF2_ITERS=240_000
USERS_PATH=os.environ.get("BURN_SHIELD_USERS") or os.path.join(os.path.dirname(os.path.abspath(__file__)),"burn_shield_users.json")
USERNAME_RE=re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
SAFE_ID=re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._\-/]{0,39}$")   # blocks =,+,-,@,<,> ... at the start / anywhere
CTRL_CHARS=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
FORMULA_START=("=","+","-","@","\t","\r")
COMMON_PASSWORDS={"password","password1","password123","123456789","1234567890","qwertyuiop","letmein123","admin12345","welcome123","iloveyou123"}

# ---------- DETECTION SETTINGS ----------
MIN_LOT=8                 # lots smaller than this are compared with the whole dataset instead
SIGNAL_CUTOFF=3.5         # classic robust-Z outlier cutoff, used to explain individual signals
SENSITIVITY={"Sensitive":(4.0,8.0),"Balanced":(5.0,10.0),"Strict":(6.0,12.0)}   # (Watch, Critical) score thresholds
DEFAULT_THRESH=SENSITIVITY["Balanced"]
# (z column, raw column, readable name, short label)
SIGNALS=[("Z_Baseline","Value_0h","starting (0h) value","Baseline"),
         ("Z_Drift_0_24","Drift_0_to_24","drift during the first 24h","Early drift"),
         ("Z_Drift_24_96","Rate_24_96","drift per 24h between 24h and 96h","Mid drift"),
         ("Z_Drift_96_168","Rate_96_168","drift per 24h between 96h and 168h","Late drift"),
         ("Z_Acceleration","Acceleration","change in drift speed (24-96h versus first 24h)","Acceleration"),
         ("Z_Prediction","Residual_168h","168h value versus what the model expected","168h surprise")]

def esc(x):
    """HTML-escape anything that came from the user before it is placed inside unsafe_allow_html markup."""
    return html.escape(str(x),quote=True)

def neutralise_cell(v):
    """Stop spreadsheet formula injection (=cmd|..., +SUM(), @...) in exported text cells."""
    if isinstance(v,str):
        v=CTRL_CHARS.sub("",v)
        if v.startswith(FORMULA_START): return "'"+v
    return v

def to_safe_csv(df):
    out=df.copy()
    for c in out.columns:
        if out[c].dtype==object or str(out[c].dtype).startswith("str"):
            out[c]=out[c].map(neutralise_cell)
    return out.to_csv(index=False).encode("utf-8")

# ---------- LOGIN SYSTEM (accounts + password) ----------
@st.cache_resource
def _store_lock():
    return threading.Lock()

def _load_users():
    try:
        with open(USERS_PATH,"r",encoding="utf-8") as f: data=json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(data,dict): raise RuntimeError("user store is corrupt")
    return data

def _save_users(users):
    tmp=USERS_PATH+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(users,f)
    try: os.chmod(tmp,0o600)
    except OSError: pass
    os.replace(tmp,USERS_PATH)

def hash_password(pw,salt=None,iters=PBKDF2_ITERS):
    salt=salt or os.urandom(16)
    h=hashlib.pbkdf2_hmac("sha256",pw.encode("utf-8"),salt,iters)
    return {"salt":salt.hex(),"hash":h.hex(),"iters":iters}

def check_password(pw,rec):
    h=hashlib.pbkdf2_hmac("sha256",pw.encode("utf-8"),bytes.fromhex(rec["salt"]),int(rec["iters"]))
    return hmac.compare_digest(h.hex(),rec["hash"])

_DUMMY={"salt":"00"*16,"hash":"00"*32,"iters":PBKDF2_ITERS}   # equalises timing when the username does not exist

def password_problems(pw,username=""):
    p=[]
    if len(pw)<10: p.append("at least 10 characters")
    if sum(bool(re.search(x,pw)) for x in (r"[a-z]",r"[A-Z]",r"\d",r"[^A-Za-z0-9]"))<3:
        p.append("at least 3 of: lowercase, UPPERCASE, number, symbol")
    if username and username.lower() in pw.lower(): p.append("must not contain your username")
    if pw.lower() in COMMON_PASSWORDS: p.append("must not be a common password")
    return p

def register_user(username,pw_record):
    """Returns an error string, or None on success."""
    key=username.strip().lower()
    with _store_lock():
        users=_load_users()
        if key in users: return "That username is already taken."
        rec={"display":username.strip(),"pw":pw_record,"created":time.time(),"fails":0,"locked_until":0}
        users[key]=rec; _save_users(users)
    return None

def authenticate(username,pw):
    """Returns (ok, display_name_or_error_message)."""
    key=(username or "").strip().lower(); now=time.time()
    with _store_lock():
        users=_load_users(); rec=users.get(key)
        if rec and now<rec.get("locked_until",0):
            return False,f"Account temporarily locked after too many failed attempts. Try again in {int((rec['locked_until']-now)//60)+1} minute(s)."
        pw_ok=check_password(pw,rec["pw"] if rec else _DUMMY)
        ok=bool(rec) and pw_ok
        if ok:
            rec["fails"]=0; rec["locked_until"]=0; users[key]=rec; _save_users(users)
            return True,rec.get("display",key)
        if rec:
            rec["fails"]=rec.get("fails",0)+1
            if rec["fails"]>=MAX_LOGIN_TRIES: rec["locked_until"]=now+LOCKOUT_SEC; rec["fails"]=0
            users[key]=rec; _save_users(users)
        log.warning("Failed login attempt")
        return False,"Invalid username or password."

def auth_gate():
    """First page: login / create account. Returns the signed-in username (or stops the script)."""
    if st.session_state.get("auth_user"):
        now=time.time()
        if now-st.session_state.get("last_active",now)>SESSION_TIMEOUT:
            st.session_state.clear(); st.session_state.timed_out=True; st.rerun()
        st.session_state.last_active=now
        return st.session_state.auth_user
    st.markdown('<div class="hero center"><span class="badge">SECURE ACCESS</span><h1>🛡️ BURN-SHIELD</h1><p>Sign in to continue. Accounts are protected with salted password hashing and account lockout.</p></div>',unsafe_allow_html=True)
    if st.session_state.pop("timed_out",False): st.warning("You were signed out after 30 minutes of inactivity.")
    st.write("")
    _,mid,_=st.columns([1,1.4,1])
    with mid:
        tab_login,tab_reg=st.tabs(["🔑 Login","📝 Create account"])
        with tab_login:
            with st.form("login_form"):
                u=st.text_input("Username",max_chars=30)
                p=st.text_input("Password",type="password",max_chars=128)
                go_login=st.form_submit_button("Login",use_container_width=True)
            if go_login:
                if not u or not p: st.error("Enter your username and password.")
                else:
                    try: ok,msg=authenticate(u,p)
                    except Exception:
                        log.exception("Login failure"); ok,msg=False,"Login is temporarily unavailable. Please try again."
                    if ok:
                        st.session_state.auth_user=msg; st.session_state.last_active=time.time(); st.rerun()
                    else:
                        time.sleep(.7); st.error(msg)
        with tab_reg:
            invite_required=os.environ.get("BURN_SHIELD_INVITE")
            if st.session_state.pop("reg_done",False): st.success("Account created. Open the Login tab and sign in.")
            with st.form("reg_form"):
                nu=st.text_input("Choose a username",max_chars=30,help="3-30 characters: letters, numbers, . _ -")
                np1=st.text_input("Choose a password",type="password",max_chars=128,help="At least 10 characters, with 3 of: lowercase, UPPERCASE, number, symbol.")
                np2=st.text_input("Confirm password",type="password",max_chars=128)
                inv=st.text_input("Invite code",type="password",max_chars=64) if invite_required else ""
                go_reg=st.form_submit_button("Create account",use_container_width=True)
            if go_reg:
                errs=[]
                if not USERNAME_RE.match(nu or ""): errs.append("Username must be 3-30 characters: letters, numbers, . _ -")
                if np1!=np2: errs.append("Passwords do not match.")
                errs+= ["Password needs: "+x for x in password_problems(np1,nu or "")]
                if invite_required and not hmac.compare_digest((inv or "").encode(),invite_required.encode()): errs.append("Invalid invite code.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    try:
                        err=register_user(nu,hash_password(np1))
                        if err: st.error(err)
                        else: st.session_state.reg_done=True; st.rerun()
                    except Exception:
                        log.exception("Registration failure"); st.error("Could not create the account. Please try again.")
        st.caption("Passwords are never stored in plain text. Too many wrong attempts lock the account for 5 minutes.")
    st.stop()

def allow_run():
    """Simple per-session rate limit for the analysis."""
    now=time.time()
    runs=[t for t in st.session_state.get("run_times",[]) if now-t<60]
    if len(runs)>=MAX_RUNS_PER_MIN: return False
    runs.append(now); st.session_state.run_times=runs
    return True

def _bad_examples(series,n=3):
    return ", ".join(repr(str(v)[:20]) for v in series.head(n))

def safe_read_csv(up):
    """Validate and read an uploaded CSV defensively. Returns (dataframe, info). Raises ValueError with a clear message."""
    name=os.path.basename(up.name or "upload.csv")
    if not name.lower().endswith(".csv"): raise ValueError("Only .csv files are accepted.")
    if up.size>MAX_UPLOAD_MB*1024*1024: raise ValueError(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    blob=up.getvalue()
    if not blob.strip(): raise ValueError("The file is empty.")
    if b"\x00" in blob[:65536]: raise ValueError("This does not look like a plain-text CSV (binary content detected).")
    try: text=blob.decode("utf-8-sig")
    except UnicodeDecodeError: text=blob.decode("latin-1")
    try:
        raw=pd.read_csv(io.StringIO(text),dtype=str,nrows=MAX_ROWS+1,skipinitialspace=True)
    except Exception:
        raise ValueError("The CSV could not be parsed. Check that it is a comma-separated file with a header row.")
    if raw.shape[1]>MAX_COLS: raise ValueError(f"Too many columns (limit {MAX_COLS}).")
    if len(raw)>MAX_ROWS: raise ValueError(f"Too many rows (limit {MAX_ROWS:,}).")
    cols_in=raw.shape[1]; rows_in=len(raw)
    # tolerant, case-insensitive header matching -> canonical names
    canon={c.lower().replace(" ","_"):c for c in [*REQUIRED,*OPTIONAL]}
    raw.columns=[canon.get(str(c).strip().lower().replace(" ","_"),str(c).strip()) for c in raw.columns]
    if raw.columns.duplicated().any(): raise ValueError("Duplicate column names found in the header.")
    known=[c for c in [*REQUIRED,*OPTIONAL] if c in raw.columns]
    missing=[c for c in REQUIRED if c not in raw.columns]
    if missing: raise ValueError("Missing required columns: "+", ".join(missing))
    raw=raw[known].copy()                      # unknown columns are dropped (smaller attack surface)
    for c in ["Component_ID","Lot_ID"]:
        if c in raw.columns:
            raw[c]=raw[c].fillna("").astype(str).map(lambda v: CTRL_CHARS.sub("",v).strip())
            if c=="Lot_ID": raw[c]=raw[c].replace("","UNKNOWN")
            bad=raw[~raw[c].str.match(SAFE_ID)][c]
            if len(bad):
                raise ValueError(f"{len(bad)} invalid value(s) in {c}, e.g. {_bad_examples(bad)}. "
                                 "Allowed: letters, numbers, space, . _ - / (max 40 characters, must start with a letter or number).")
    info={"rows_in":rows_in,"cols_in":cols_in,"cols_ignored":cols_in-len(known),"size_kb":up.size/1024}
    return raw,info

# ---------- DEMO DATA ----------
def make_demo(seed=42,n=120):
    rng=np.random.default_rng(seed); rows=[]
    anomaly_ids={7:"Early jump",23:"Accelerating drift",61:"Late jump",88:"Gradual drift",101:"Shifted baseline"}
    for i in range(n):
        lot=f"L{(i//30)+1:02d}"
        base=rng.normal(10+0.35*((i//30)%3),.28)
        drift=rng.normal(.65,.09)
        v0=base+rng.normal(0,.06)
        v24=v0+drift+rng.normal(0,.05)
        v96=v24+1.7*drift+rng.normal(0,.07)
        v168=v96+1.8*drift+rng.normal(0,.08)
        label=0
        if i in anomaly_ids:
            label=1
            if i==7: v24=v0+2.4; v96=v24+1.7; v168=v96+2.0
            elif i==23: v24=v0+.8; v96=v24+3.0; v168=v96+4.0
            elif i==61: v24=v0+.6; v96=v24+.9; v168=v96+3.2
            elif i==88: v24=v0+1.0; v96=v24+1.8; v168=v96+2.8
            else: v0+=2.0; v24=v0+.7; v96=v24+1.2; v168=v96+1.3
        rows.append([f"C-{i+1:03d}",lot,v0,v24,v96,v168,label])
    return pd.DataFrame(rows,columns=["Component_ID","Lot_ID",*VALUE_COLS,"Anomaly_Label"])

# ---------- DATA + MODEL ----------
def robust_z(s):
    """Median/MAD z-score (falls back to standard deviation when MAD is zero)."""
    s=pd.Series(s)
    med=s.median(); mad=np.median(np.abs(s-med))
    if mad>1e-9: return .6745*(s-med)/mad
    sd=s.std()
    if np.isfinite(sd) and sd>1e-9: return (s-med)/sd
    return pd.Series(np.zeros(len(s)),index=s.index)

def lot_z(d,col):
    """Robust Z of `col` inside each production lot; tiny lots are compared with the whole dataset."""
    size=d.groupby("Lot_ID")[col].transform("size")
    zl=d.groupby("Lot_ID")[col].transform(robust_z)
    zg=robust_z(d[col])
    return pd.Series(np.where(size>=MIN_LOT,zl,zg),index=d.index).replace([np.inf,-np.inf],np.nan).fillna(0)

def prepare(d,report=None):
    """Validate + clean. Fills `report` (dict) with what was found/removed so it can be explained to the user."""
    rep=report if report is not None else {}
    required={"Component_ID",*VALUE_COLS}
    missing=required-set(d.columns)
    if missing: raise ValueError("Missing required columns: "+", ".join(sorted(missing)))
    d=d.copy()
    rep.setdefault("rows_in",len(d)); rep.setdefault("cols_in",d.shape[1]); rep.setdefault("cols_ignored",0)
    rep["lot_supplied"]="Lot_ID" in d.columns
    if not rep["lot_supplied"]: d["Lot_ID"]="LOT-01"
    for c in VALUE_COLS: d[c]=pd.to_numeric(d[c],errors="coerce")
    d[VALUE_COLS]=d[VALUE_COLS].replace([np.inf,-np.inf],np.nan)
    n0=len(d)
    d=d.dropna(subset=VALUE_COLS); rep["bad_rows"]=n0-len(d)
    n1=len(d)
    d=d[~(d[VALUE_COLS].abs()>MAX_ABS_VALUE).any(axis=1)]; rep["range_rows"]=n1-len(d)
    n2=len(d)
    d=d.drop_duplicates(subset="Component_ID",keep="first"); rep["dup_rows"]=n2-len(d)
    rep["label_used"]=False
    if "Anomaly_Label" in d.columns:
        lab=pd.to_numeric(d["Anomaly_Label"],errors="coerce")
        if lab.isin([0,1]).all(): d["Anomaly_Label"]=lab.astype(int); rep["label_used"]=True
        else: d=d.drop(columns=["Anomaly_Label"]); rep["label_dropped"]=True
    d=d.reset_index(drop=True)
    rep["rows_final"]=len(d)
    if len(d)<MIN_ROWS: raise ValueError(f"Only {len(d)} valid component(s) remain after cleaning; at least {MIN_ROWS} are needed.")
    return d

def analyze(d,thresholds=DEFAULT_THRESH):
    """Lot-relative multi-signal anomaly scoring. Returns (result, model, mae, analysis_summary)."""
    watch_t,crit_t=thresholds
    d=d.copy()
    # --- drift features (rates are per 24h so different time windows are comparable) ---
    d["Drift_0_to_24"]=d.Value_24h-d.Value_0h
    d["Drift_24_to_96"]=d.Value_96h-d.Value_24h
    d["Drift_96_to_168"]=d.Value_168h-d.Value_96h
    d["Rate_24_96"]=d.Drift_24_to_96/3
    d["Rate_96_168"]=d.Drift_96_to_168/3
    d["Acceleration"]=d.Rate_24_96-d.Drift_0_to_24

    # --- honest 168h prediction: every component is predicted by a model that never saw it (cross-validation) ---
    Xp=d[["Value_0h","Value_24h","Value_96h"]]; y=d.Value_168h
    mk=lambda: RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=42)
    folds=min(5,len(d))
    d["Predicted_168h"]=cross_val_predict(mk(),Xp,y,cv=KFold(folds,shuffle=True,random_state=42))
    d["Residual_168h"]=y-d.Predicted_168h
    mae=float(np.mean(np.abs(d.Residual_168h)))
    model=mk().fit(Xp,y)
    importance=pd.DataFrame({"Input":["Value_0h","Value_24h","Value_96h"],"Importance":model.feature_importances_.round(3)}).sort_values("Importance",ascending=False)

    # --- lot-relative robust signals ---
    Z=pd.DataFrame({z:lot_z(d,raw) for z,raw,_,_ in SIGNALS},index=d.index)
    for z in Z.columns: d[z]=Z[z]
    A=Z.abs().clip(upper=15)
    srt=np.sort(A.values,axis=1)[:,::-1]
    evidence=.7*srt[:,0]+.3*srt[:,1]

    # --- ML outlier detector on the same signals ---
    iso=IsolationForest(n_estimators=200,random_state=42).fit(Z)
    s=-iso.score_samples(Z); ms=(s-np.median(s))/(np.std(s)+1e-9)
    d["AI_Outlier"]=(ms>=2).astype(int)
    d["Risk_Score"]=evidence+.5*np.clip(ms,0,6)
    d["Main_Signal"]=A.idxmax(axis=1).map({z:short for z,_,_,short in SIGNALS})
    d["Risk"]=np.select([d.Risk_Score>=crit_t,d.Risk_Score>=watch_t],["Critical","Watch"],default="Normal")

    lots=d.groupby("Lot_ID").agg(Components=("Component_ID","count"),Median_0h=("Value_0h","median"),Median_24h=("Value_24h","median"),
                                 Median_96h=("Value_96h","median"),Median_168h=("Value_168h","median"),
                                 Typical_24h_drift=("Drift_0_to_24","median"),Drift_spread=("Drift_0_to_24","std")).reset_index().round(3)
    sig=pd.DataFrame({"Signal":[nm for _,_,nm,_ in SIGNALS],"Components beyond ±3.5":[int((d[z].abs()>=SIGNAL_CUTOFF).sum()) for z,_,_,_ in SIGNALS]})
    ana={"lots":lots,"signals":sig,"iso":int(d.AI_Outlier.sum()),"importance":importance,"folds":folds,"mae":mae,
         "watch_t":float(watch_t),"crit_t":float(crit_t),
         "mean_d1":float(d.Drift_0_to_24.mean()),"mean_d2":float((d.Drift_24_to_96/3).mean()),"mean_d3":float((d.Drift_96_to_168/3).mean())}
    return d,model,mae,ana

# ---------- EXPLANATIONS (one by one) ----------
def reference_group(d,row):
    lot=d[d.Lot_ID==row.Lot_ID]
    return (lot,"lot") if len(lot)>=MIN_LOT else (d,"whole dataset")

def explain_component(row,ref,scope="lot"):
    """Plain-language reasons with real numbers for one component."""
    r=[]
    pairs=sorted([(abs(row[z]),z,raw,nm) for z,raw,nm,_ in SIGNALS],reverse=True)
    for az,z,raw,nm in pairs:
        if az<SIGNAL_CUTOFF: continue
        zv=row[z]
        if z=="Z_Prediction":
            r.append(f"**168h value is a surprise:** actual {row.Value_168h:.2f} but the model expected about {row.Predicted_168h:.2f} from the 0h/24h/96h readings (gap {row.Residual_168h:+.2f}, robust Z = {zv:+.1f}).")
        else:
            r.append(f"**{nm[0].upper()+nm[1:]}** is {row[raw]:.2f} while the {scope} median is {ref[raw].median():.2f} - {'higher' if zv>0 else 'lower'} than its peers (robust Z = {zv:+.1f}, alert line ±{SIGNAL_CUTOFF}).")
    if row.AI_Outlier: r.append("**ML outlier:** the Isolation Forest finds this combination of signals unusual compared with the other components.")
    if not r:
        az,z,raw,nm=pairs[0]
        r.append(f"All signals are within the normal range of its {scope}. The strongest one is the {nm} (robust Z = {row[z]:+.1f}, below ±{SIGNAL_CUTOFF}).")
    return r

def build_steps(d,ana,q):
    nl=d.Lot_ID.nunique(); top=d.sort_values("Risk_Score",ascending=False).iloc[0]
    cnt={k:int((d.Risk==k).sum()) for k in ["Normal","Watch","Critical"]}
    s=[]
    f=[f"**{q.get('rows_in',len(d))}** rows and **{q.get('cols_in',d.shape[1])}** columns were read."]
    f.append(f"**{nl}** production lot(s) found in `Lot_ID`." if q.get("lot_supplied") else "No `Lot_ID` column, so all components were treated as one lot (`LOT-01`).")
    f.append("`Anomaly_Label` found - it is used only to score the result afterwards, never to detect anomalies." if q.get("label_used") else "No `Anomaly_Label` column - that is fine, it is only needed to compute accuracy.")
    if q.get("cols_ignored"): f.append(f"{q['cols_ignored']} extra column(s) were ignored for safety.")
    s.append(dict(title="Reading measurements",headline=f"{q.get('rows_in',len(d))} rows, {nl} lot(s)",
        did="Opened the file and identified each component and its four measurements (0h, 24h, 96h, 168h).",found=f,
        meaning="Each row is one component. The four numbers describe its burn-in trajectory, which is what everything else is built on.",tables=[]))
    f=[f"**{q.get('bad_rows',0)}** row(s) removed because a measurement was missing or not a number.",
       f"**{q.get('range_rows',0)}** row(s) removed because of impossible/extreme values.",
       f"**{q.get('dup_rows',0)}** duplicate `Component_ID` row(s) removed (first one kept).",
       f"**{len(d)}** clean components continue to the analysis."]
    if q.get("label_dropped"): f.append("`Anomaly_Label` was ignored because it contained values other than 0 and 1.")
    s.append(dict(title="Validating data quality",headline=f"{len(d)} clean components",
        did="Converted measurements to numbers, removed unusable rows and checked IDs for duplicates.",found=f,
        meaning="Nothing is guessed or filled in. Only complete, numeric data reaches the models, so results reflect your file.",tables=[]))
    s.append(dict(title="Comparing lot behaviour",headline=f"{nl} lot(s) profiled",
        did="Computed the typical (median) value of each lot at every time point. Every signal is then converted to a robust Z-score (median/MAD) inside the component's own lot. Lots with fewer than 8 components are compared with the whole dataset.",
        found=["Robust Z = how many 'robust standard deviations' a component is from its lot's typical value. Beyond ±3.5 is a recognised outlier cutoff."],
        meaning="Components are judged against their own production lot, so a lot that normally runs higher is not unfairly flagged. Median/MAD is used because a few bad parts cannot distort it.",
        tables=[("Lot profile (medians)",ana["lots"])]))
    s.append(dict(title="Detecting abnormal drift",headline=f"{int((ana['signals'].iloc[1:5,1]>0).sum())} drift signals active, {ana['iso']} ML outliers",
        did="Measured drift in each window as a rate per 24h (0-24h, 24-96h, 96-168h) so different window lengths are comparable, plus acceleration = drift speed in 24-96h minus drift speed in the first 24h. An Isolation Forest then looked at all signals together.",
        found=[f"Average drift per 24h: **{ana['mean_d1']:+.2f}** (0-24h), **{ana['mean_d2']:+.2f}** (24-96h), **{ana['mean_d3']:+.2f}** (96-168h).",
               f"The Isolation Forest marked **{ana['iso']}** component(s) as unusual combinations."],
        meaning="A healthy part drifts in line with its lot. Parts whose drift speed jumps, accelerates or differs from peers at any stage are the early-warning candidates. The table shows how many components break each signal.",
        tables=[("Components beyond the ±3.5 alert line, per signal",ana["signals"])]))
    s.append(dict(title="Predicting future trajectory",headline=f"MAE {ana['mae']:.3f}",
        did=f"Trained a Random Forest on 0h, 24h and 96h readings to predict the 168h value, using {ana['folds']}-fold cross-validation: every component is predicted by a model that never saw it.",
        found=[f"Average prediction error (MAE) on unseen components: **{ana['mae']:.3f}** measurement units."],
        meaning="Because predictions are made out-of-sample, a big gap between the actual and expected 168h value is real evidence of unusual behaviour (the '168h surprise' signal), not a model that memorised the part.",
        tables=[("What the prediction relied on",ana["importance"])]))
    s.append(dict(title="Generating component risk profile",headline=f"{cnt['Critical']} critical, {cnt['Watch']} watch",
        did="Risk Score = 70% strongest signal + 30% second-strongest signal (robust Z units) + a small bonus when the ML detector also finds the part unusual.",
        found=[f"Watch = score ≥ **{ana['watch_t']:.1f}**, Critical = score ≥ **{ana['crit_t']:.1f}**.",
               f"Result: **{cnt['Critical']}** Critical, **{cnt['Watch']}** Watch, **{cnt['Normal']}** Normal.",
               f"Highest score: **{top.Component_ID}** (lot {top.Lot_ID}, score {top.Risk_Score:.2f}, main signal: {top.Main_Signal})."],
        meaning="Risk is based on the strength of evidence, not a fixed percentage - a clean dataset can legitimately have zero flagged components. Critical means 'inspect first', not a confirmed failure.",
        tables=[]))
    return s

def perf_block(y,pred,risk=None):
    """Accuracy / Precision / Recall / F1 plus the confusion matrix."""
    q=st.columns(4)
    for c,lab,val in zip(q,["Accuracy","Precision","Recall","F1"],[accuracy_score(y,pred),precision_score(y,pred,zero_division=0),recall_score(y,pred,zero_division=0),f1_score(y,pred,zero_division=0)]):
        c.metric(lab,f"{val:.1%}")
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    st.caption(f"Caught {tp} of {tp+fn} labelled anomalies · {fp} false alarm(s) · {fn} missed · {tn} correctly left Normal.")
    if risk is not None and (risk=="Critical").any():
        crit=(risk=="Critical").astype(int)
        st.caption(f"Critical tier alone: precision {precision_score(y,crit,zero_division=0):.1%}, recall {recall_score(y,crit,zero_division=0):.1%}.")

# ---------- STATE ----------
def reset():
    for k in ["data","model","mae","filename","ana","qrep","steps","raw_ok","thresholds"]:
        st.session_state.pop(k,None)
    st.session_state.page="analyze"

if "page" not in st.session_state: st.session_state.page="home"

# ---------- LOGIN (first page) ----------
current_user=auth_gate()

# ---------- SIDEBAR ----------
NAV=["Home","How to Use","Demo","Analyze"]
PAGE_TO_NAV={"home":0,"how to use":1,"demo":2,"analyze":3}
with st.sidebar:
    st.markdown("## 🛡️ BURN-SHIELD")
    st.caption("Burn-In Intelligence")
    st.markdown(f"Signed in as **{esc(current_user)}**")
    if st.button("Log out",use_container_width=True):
        st.session_state.clear(); st.rerun()
    st.divider()
    choice=st.radio("Navigate",NAV,index=PAGE_TO_NAV.get(st.session_state.page,3))
    st.divider()
    st.markdown("**Workflow**")
    st.caption("Upload → Validate → Compare → Detect → Predict → Explain")
    st.divider()
    with st.expander("🔒 Security"):
        st.markdown(f"""- Login required, salted PBKDF2 password hashing
- Account lock after {MAX_LOGIN_TRIES} wrong attempts, auto logout after 30 min idle
- CSV only, max {MAX_UPLOAD_MB} MB, {MAX_ROWS:,} rows
- Binary / malformed files rejected
- IDs checked; formula & HTML injection blocked
- Exports are formula-safe
- Data kept in memory for this session only
- Analysis rate-limited per session""")
    st.caption("Standalone software prototype")
    if st.session_state.get("data") is not None and st.session_state.page=="results":
        if st.button("＋ New Analysis",use_container_width=True):
            reset(); st.rerun()

newpage={"Home":"home","How to Use":"how to use","Demo":"demo","Analyze":"analyze"}[choice]
if st.session_state.page not in ["results","analyzing"] and newpage!=st.session_state.page:
    st.session_state.page=newpage

# ---------- HOME ----------
if st.session_state.page=="home":
    st.markdown('<div class="hero center"><span class="badge">BURN-IN INTELLIGENCE</span><h1>🛡️ BURN-SHIELD</h1><p>Analyze burn-in measurements over time, compare component behaviour with its production lot, detect abnormal drift, and estimate future trajectory.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="flow"><div class="flowbox"><b>01 · Observe</b><span>Read burn-in measurements</span></div><div class="flowbox"><b>02 · Compare</b><span>Learn lot behaviour</span></div><div class="flowbox"><b>03 · Detect</b><span>Find unusual drift</span></div><div class="flowbox"><b>04 · Predict</b><span>Estimate trajectory</span></div></div>',unsafe_allow_html=True)
    a,b,c=st.columns([1,1,1])
    with b:
        if st.button("Start Analysis →",use_container_width=True): st.session_state.page="analyze"; st.rerun()
    st.markdown('<div class="card"><h3>What is BURN-SHIELD?</h3><p class="muted">A software prototype for analysing time-series burn-in data. It does not physically test components. It turns measurements into an engineer-friendly screening and investigation view.</p></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card"><h3>What it looks for</h3><p class="muted">Lot-relative deviation in the starting value, early/mid/late drift speed, acceleration, surprises in the final reading and machine-learning outlier signals.</p></div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>What you get</h3><p class="muted">A step-by-step explanation of every analysis stage, a results dashboard where you can pick Critical, Watch or Normal components, each explained one by one, trajectory charts and accuracy metrics.</p></div>',unsafe_allow_html=True)
    st.stop()

# ---------- HOW TO USE ----------
if st.session_state.page=="how to use":
    st.markdown('<div class="hero"><span class="badge">QUICK GUIDE</span><h2>How to use BURN-SHIELD</h2><p>Start here if you are opening the project for the first time.</p></div>',unsafe_allow_html=True)
    instructions=[
        ("01","Prepare your CSV","Minimum columns: Component_ID, Value_0h, Value_24h, Value_96h, Value_168h. Lot_ID is recommended but optional. At least 12 components are needed."),
        ("02","Open Analyze","Use the left sidebar and choose Analyze. The upload box appears only on this page."),
        ("03","Pick a sensitivity and upload","Choose Sensitive, Balanced or Strict, upload your CSV and press Analyze Dataset. The file is security-checked first."),
        ("04","Watch the analysis","Each stage (read, validate, compare lots, detect drift, predict, risk profile) completes one by one with what it found in your data."),
        ("05","Choose what to inspect","On the results page select Critical, Watch and/or Normal. The table, the one-by-one explanations and the investigation chart follow your choice."),
        ("06","Check performance","If your CSV contains Anomaly_Label (0/1), the dashboard calculates Accuracy, Precision, Recall and F1 against those supplied labels.")
    ]
    for n,t,dsc in instructions:
        st.markdown(f'<div class="step"><b>{n} · {t}</b><br><span class="muted">{dsc}</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>CSV example</h3></div>',unsafe_allow_html=True)
    st.dataframe(make_demo().head(6)[["Component_ID","Lot_ID",*VALUE_COLS]],use_container_width=True,hide_index=True)
    st.stop()

# ---------- DEMO ----------
if st.session_state.page=="demo":
    st.markdown('<div class="hero"><span class="badge">DEMO MODE</span><h2>See the complete workflow before uploading your own data</h2><p>This page uses built-in synthetic burn-in data so you can understand what the application does without preparing a CSV first.</p></div>',unsafe_allow_html=True)
    demo=make_demo()
    st.markdown('<div class="section">1 · Example input</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><p class="muted">Each row represents a component. The four measurement columns represent observations at 0h, 24h, 96h and 168h. The demo also contains hidden ground-truth labels so model metrics can be shown.</p></div>',unsafe_allow_html=True)
    st.dataframe(demo.head(10),use_container_width=True,hide_index=True)
    st.markdown('<div class="section">2 · Run the same analysis</div>',unsafe_allow_html=True)
    demo_result,_,demo_mae,_=analyze(prepare(demo))
    m=st.columns(5)
    for c,(lab,val) in zip(m,[("Components",len(demo_result)),("Normal",(demo_result.Risk=="Normal").sum()),("Watch",(demo_result.Risk=="Watch").sum()),("Critical",(demo_result.Risk=="Critical").sum()),("MAE",f"{demo_mae:.2f}")]):
        c.markdown(f'<div class="metric"><div class="label">{lab}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">3 · What the result means</div>',unsafe_allow_html=True)
    top=demo_result.sort_values("Risk_Score",ascending=False).head(8)
    st.dataframe(top[["Component_ID","Lot_ID","Risk","Risk_Score","Main_Signal","Acceleration","Predicted_168h"]],use_container_width=True,hide_index=True)
    y=demo_result.Anomaly_Label.astype(int); pred=(demo_result.Risk!="Normal").astype(int)
    st.markdown('<div class="card"><h3>Demo model metrics</h3><p class="muted">These are calculated against the synthetic labels included only for this demonstration. They are not real-world validation.</p></div>',unsafe_allow_html=True)
    perf_block(y,pred,demo_result.Risk)
    if st.button("Use this demo flow → Analyze",use_container_width=True):
        st.session_state.data=demo.drop(columns=["Anomaly_Label"])
        st.session_state.filename="Built-in demo dataset"
        st.session_state.qrep={"rows_in":len(demo),"cols_in":demo.shape[1]-1,"cols_ignored":0}
        st.session_state.thresholds=DEFAULT_THRESH
        st.session_state.page="analyzing"
        st.rerun()
    st.stop()

# ---------- ANALYZE UPLOAD ----------
if st.session_state.page=="analyze":
    st.markdown('<div class="hero center"><span class="badge">DATA ANALYSIS</span><h2>Upload your burn-in dataset</h2><p>Upload a CSV to start a fresh BURN-SHIELD analysis. This is the only page where the upload control appears.</p></div>',unsafe_allow_html=True)
    st.write("")
    sens=st.radio("Detection sensitivity",list(SENSITIVITY),index=1,horizontal=True,
                  help="Sensitive catches more subtle cases (more false alarms). Strict flags only strong cases (fewer false alarms). Balanced is recommended.")
    up=st.file_uploader("Choose CSV",type=["csv"],help=f"Required: Component_ID, Value_0h, Value_24h, Value_96h, Value_168h. Lot_ID is optional. Max {MAX_UPLOAD_MB} MB.")
    if up:
        st.markdown(f'<div class="card"><h3>Ready to analyze</h3><p class="muted">{esc(os.path.basename(up.name))} · {up.size/1024:.1f} KB</p></div>',unsafe_allow_html=True)
        if st.button("Analyze Dataset →",use_container_width=True):
            if not allow_run():
                st.error("Too many analyses in a short time. Please wait a minute and try again.")
            else:
                try:
                    raw,info=safe_read_csv(up)
                    prepare(raw,dict(info))          # early validation so errors show here
                    st.session_state.data=raw
                    st.session_state.qrep=info
                    st.session_state.thresholds=SENSITIVITY[sens]
                    st.session_state.filename=os.path.basename(up.name)
                    st.session_state.page="analyzing"
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception:
                    log.exception("Unexpected error while reading upload")
                    st.error("The file could not be processed. Please check the format and try again.")
    st.markdown('<div class="card"><h3>🔒 Your file is checked before analysis</h3><p class="muted">CSV only · size and row limits · binary content rejected · IDs validated · formula/HTML injection blocked · processed in memory only, never stored.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Need an example first?</h3><p class="muted">Open <b>Demo</b> in the left sidebar. It explains the input, runs the analysis on synthetic data and shows what the output means.</p></div>',unsafe_allow_html=True)
    st.stop()

# ---------- ANALYSIS SCREEN ----------
if st.session_state.page=="analyzing":
    st.markdown('<div class="hero center"><span class="badge">ANALYSIS IN PROGRESS</span><h2>Analyzing your burn-in data…</h2><p>Each stage is completed below together with what it found in your file.</p></div>',unsafe_allow_html=True)
    try:
        qrep=dict(st.session_state.get("qrep",{}))
        clean=prepare(st.session_state.data,qrep)
        res,model,mae,ana=analyze(clean,st.session_state.get("thresholds",DEFAULT_THRESH))
        steps=build_steps(res,ana,qrep)
    except ValueError as e:
        st.error("Analysis could not be completed: "+str(e))
        if st.button("Back to Analyze"): reset(); st.rerun()
        st.stop()
    except Exception:
        log.exception("Unexpected analysis error")
        st.error("Analysis could not be completed because of an unexpected error. Your data was not stored.")
        if st.button("Back to Analyze"): reset(); st.rerun()
        st.stop()

    holder=st.empty(); progress=st.progress(0)
    for i in range(len(steps)+1):
        html_out=""
        for j,s in enumerate(steps):
            if j<i:
                html_out+=f'<div class="step done"><b>✓ {esc(s["title"])}</b><br><span class="muted">{esc(s["headline"])} - {esc(s["did"])}</span></div>'
            elif j==i:
                html_out+=f'<div class="step active"><b>◉ {esc(s["title"])}…</b><br><span class="muted">{esc(s["did"])}</span></div>'
            else:
                html_out+=f'<div class="step"><b>○ {esc(s["title"])}</b></div>'
        holder.markdown(html_out,unsafe_allow_html=True)
        progress.progress(int(min(i,len(steps))/len(steps)*100))
        time.sleep(.6)
    st.session_state.data,st.session_state.model,st.session_state.mae=res,model,mae
    st.session_state.ana=ana; st.session_state.steps=steps
    st.session_state.page="results"
    st.rerun()

# ---------- RESULTS ----------
if st.session_state.page=="results":
    d=st.session_state.data
    mae=st.session_state.get("mae",np.nan)
    st.markdown(f'<div class="hero"><span class="badge">ANALYSIS COMPLETE</span><h2>Burn-in analysis results</h2><p>Dataset: {esc(st.session_state.get("filename",""))}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Screening overview</div>',unsafe_allow_html=True)
    metrics=[("Components",len(d)),("Lots",d.Lot_ID.nunique()),("Normal",(d.Risk=="Normal").sum()),("Watch",(d.Risk=="Watch").sum()),("Critical",(d.Risk=="Critical").sum())]
    cc=st.columns(5)
    for c,(lab,val) in zip(cc,metrics):
        c.markdown(f'<div class="metric"><div class="label">{lab}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)

    # ----- step-by-step explanation -----
    st.markdown('<div class="section">Step-by-step: what BURN-SHIELD did with your data</div>',unsafe_allow_html=True)
    st.caption("Every number below was calculated from your uploaded file.")
    for i,s in enumerate(st.session_state.get("steps",[]),1):
        with st.expander(f"Step {i} · {s['title']}  —  {s['headline']}",expanded=True):
            st.markdown("**What was done**  \n"+s["did"])
            st.markdown("**What was found**")
            for line in s["found"]: st.markdown("- "+line)
            for cap,tbl in s.get("tables",[]):
                st.caption(cap); st.dataframe(tbl,use_container_width=True,hide_index=True)
            st.markdown("**What it means**  \n"+s["meaning"])

    # ----- NEW: choose Critical / Watch / Normal -----
    st.markdown('<div class="section">Explore components by risk level</div>',unsafe_allow_html=True)
    levels=st.multiselect("Select the risk levels to view",["Critical","Watch","Normal"],default=["Critical","Watch"])
    view=d[d.Risk.isin(levels)].sort_values("Risk_Score",ascending=False)
    left,right=st.columns([1.35,1])
    with left:
        st.markdown(f'<div class="card"><h3>{len(view)} component(s) shown</h3><p class="muted">Sorted by Risk Score. Main_Signal is the strongest reason behind the score.</p></div>',unsafe_allow_html=True)
        if len(view): st.dataframe(view[["Component_ID","Lot_ID","Risk","Risk_Score","Main_Signal","Drift_0_to_24","Drift_24_to_96","Acceleration","Predicted_168h"]],use_container_width=True,hide_index=True)
        elif not levels: st.info("Select at least one risk level above.")
        else: st.success("No components in the selected risk level(s).")
    with right:
        counts=d.Risk.value_counts().reindex(["Normal","Watch","Critical"],fill_value=0)
        fig=go.Figure(go.Bar(x=counts.index,y=counts.values,text=counts.values,textposition="auto"))
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#dbeef5"))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section">Selected components explained one by one</div>',unsafe_allow_html=True)
    if len(view)==0:
        st.info("Nothing to explain for the current selection.")
    else:
        n_show=int(st.number_input("How many components to explain (highest risk first)",min_value=1,max_value=min(40,len(view)),value=min(8,len(view)),step=1))
        icons={"Critical":"🔴","Watch":"🟠","Normal":"🟢"}
        for _,r in view.head(n_show).iterrows():
            ref,scope=reference_group(d,r)
            with st.expander(f"{icons[r.Risk]} {r.Component_ID} · {r.Risk} · score {r.Risk_Score:.2f} · lot {r.Lot_ID}"):
                st.markdown(f"**Trajectory:** {r.Value_0h:.2f} → {r.Value_24h:.2f} → {r.Value_96h:.2f} → {r.Value_168h:.2f}  \n"
                            f"**{scope.capitalize()} median:** {ref.Value_0h.median():.2f} → {ref.Value_24h.median():.2f} → {ref.Value_96h.median():.2f} → {ref.Value_168h.median():.2f}")
                st.markdown("**Why it was flagged**" if r.Risk!="Normal" else "**Why it is considered normal**")
                for line in explain_component(r,ref,scope): st.markdown("- "+line)
                if r.Risk=="Critical": st.markdown("**Suggested action:** inspect this component first and review its lot.")
                elif r.Risk=="Watch": st.markdown("**Suggested action:** keep under observation and compare with the next batch.")

    st.markdown('<div class="section">Component investigation</div>',unsafe_allow_html=True)
    if len(view)==0:
        st.info("Select a risk level above to investigate a component.")
    else:
        selected=st.selectbox("Select a component (highest risk first)",view.Component_ID.tolist())
        row=d[d.Component_ID==selected].iloc[0]; ref,scope=reference_group(d,row)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=[0,24,96],y=[row.Value_0h,row.Value_24h,row.Value_96h],mode="lines+markers",name="Observed"))
        fig.add_trace(go.Scatter(x=[96,168],y=[row.Value_96h,row.Predicted_168h],mode="lines+markers",name="Expected 168h",line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=[168],y=[row.Value_168h],mode="markers",name="Actual 168h",marker=dict(size=11,symbol="diamond")))
        fig.add_trace(go.Scatter(x=TIMEPOINTS,y=[ref[c].median() for c in VALUE_COLS],mode="lines+markers",name=f"{scope.capitalize()} median",line=dict(dash="dot")))
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#dbeef5"),xaxis_title="Burn-in time (hours)",yaxis_title="Parameter value")
        st.plotly_chart(fig,use_container_width=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Risk",row.Risk); m2.metric("24h drift",f"{row.Drift_0_to_24:.2f}"); m3.metric("Risk score",f"{row.Risk_Score:.2f}"); m4.metric("Expected 168h",f"{row.Predicted_168h:.2f}")
        st.markdown("**Why was it flagged?**" if row.Risk!="Normal" else "**Why is it considered normal?**")
        for reason in explain_component(row,ref,scope): st.markdown("- "+reason)

    st.markdown('<div class="section">Model performance</div>',unsafe_allow_html=True)
    if "Anomaly_Label" in d.columns:
        y=d.Anomaly_Label.astype(int); pred=(d.Risk!="Normal").astype(int)
        perf_block(y,pred,d.Risk)
        st.caption("These metrics use the labels supplied in the CSV (Watch + Critical counted as flagged). They are prototype metrics, not real-world validation.")
    else:
        st.info("No Anomaly_Label was supplied, so BURN-SHIELD does not invent an accuracy score.")
    st.metric("168h prediction error (MAE, cross-validated)",f"{mae:.3f}" if np.isfinite(mae) else "N/A")

    st.markdown('<div class="section">Export</div>',unsafe_allow_html=True)
    st.download_button("⬇ Download results (CSV)",data=to_safe_csv(d),file_name="burn_shield_results.csv",mime="text/csv")

    st.markdown('<div class="section">Analysis pipeline</div>',unsafe_allow_html=True)
    for i,(t,desc) in enumerate([
        ("Observe","Read measurements at 0h, 24h, 96h and 168h."),
        ("Compare","Compare a component with its production-lot behaviour using robust statistics."),
        ("Detect","Combine lot-relative baseline, drift, acceleration, 168h surprise and an ML outlier signal."),
        ("Predict","Estimate 168h from 0h, 24h and 96h with out-of-sample cross-validation."),
        ("Explain","Show the evidence behind every risk level.")
    ],1):
        st.markdown(f'<div class="step"><b>{i:02d} · {t}</b><br><span class="muted">{desc}</span></div>',unsafe_allow_html=True)
    st.caption("BURN-SHIELD is a standalone software prototype. Synthetic/demo data and prototype thresholds require validation before engineering decisions.")
    st.stop()
