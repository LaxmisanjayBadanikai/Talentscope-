import os
import time
import threading
import joblib
import numpy as np
import pandas as pd
import shap
from collections import deque
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==========================================
# 1. LOAD MODELS & METADATA
# ==========================================
print("📦 Loading TalentScope models...")
m_dir = "models"

def load_pkl(name):
    with open(os.path.join(m_dir, name), 'rb') as f:
        return joblib.load(f)

model = load_pkl('ts_model.pkl')
df = load_pkl('ts_df.pkl')
meta = load_pkl('ts_metadata.pkl')

# Unpack metadata
residual_std = meta['residual_std']
r2_te = meta['r2_te']
mae_te = meta['mae_te']
role_stats = meta['role_stats']
skill_stats = meta['skill_stats']
importance_list = meta['importance_list']
role_profiles = meta['role_profiles']

ROLES = meta['ROLES']
SENIORITY = meta['SENIORITY']
CITIES = meta['CITIES']
COMPANY_TIERS = meta['COMPANY_TIERS']
INDUSTRIES = meta['INDUSTRIES']
REMOTE_TYPES = meta['REMOTE_TYPES']
SKILLS = meta['SKILLS']

FEATURES = [
    'role_id', 'seniority_id', 'city_id', 'company_tier_id',
    'industry_id', 'remote_type_id', 'years_experience', 'skill_count',
] + SKILLS

CITY_PROBS = np.array([0.14, 0.14, 0.09, 0.06, 0.08, 0.07, 0.05, 0.05, 0.04, 0.13, 0.06, 0.03, 0.03, 0.03])
CITY_PROBS = CITY_PROBS / CITY_PROBS.sum()
INDUSTRY_PROBS = np.array([0.15, 0.10, 0.14, 0.06, 0.08, 0.22, 0.05, 0.20])
INDUSTRY_PROBS = INDUSTRY_PROBS / INDUSTRY_PROBS.sum()

ROLE_NAME_TO_ID = {name: i for i, name in enumerate(ROLES)}

print("🧠 Rebuilding SHAP explainer...")
shap_explainer = shap.TreeExplainer(model)
print("✅ App ready!")

# ==========================================
# 2. CORE LOGIC & WORKERS
# ==========================================
def build_ml_vector(payload):
    role_id   = int(payload.get('role_id', 1))
    sen_id    = int(payload.get('seniority_id', 1))
    city_id   = int(payload.get('city_id', 0))
    comp_id   = int(payload.get('company_tier_id', 1))
    ind_id    = int(payload.get('industry_id', 5))
    rem_id    = int(payload.get('remote_type_id', 0))
    years     = int(payload.get('years_experience', 3))
    skill_list = payload.get('skills', [])
    skill_set  = set(str(s).lower() for s in skill_list)
    skill_count = len(skill_set)

    row = [role_id, sen_id, city_id, comp_id, ind_id, rem_id, years, skill_count]
    for s in SKILLS:
        row.append(1.0 if s in skill_set else 0.0)
    return np.array([row], dtype=float), skill_set

def compute_skill_gap(target_role, user_skills):
    user_skills = set(s.strip().lower() for s in user_skills if s.strip())
    role_id = ROLE_NAME_TO_ID.get(target_role)
    if role_id is None: return []

    role_postings = df[df['role_id'] == role_id]
    total = len(role_postings)
    if total < 30: return []

    results = []
    for skill in SKILLS:
        if skill in user_skills: continue
        has = role_postings[role_postings[skill] == 1]
        if len(has) < 20: continue
        pct_required = len(has) / total * 100
        if pct_required < 15: continue
        without = role_postings[role_postings[skill] == 0]
        avg_with = has['salary'].mean()
        avg_without = without['salary'].mean() if len(without) > 5 else avg_with
        uplift = float(avg_with - avg_without)
        results.append({
            'skill': skill,
            'pct_required': round(pct_required, 1),
            'salary_uplift': round(uplift),
            'priority': round(pct_required * max(uplift, 0) / 1000, 1),
        })
    results.sort(key=lambda r: -r['priority'])
    return results[:10]

def career_next_steps(current_role, years_exp):
    current_id = ROLE_NAME_TO_ID.get(current_role)
    if current_id is None: return []
    cur = role_profiles[role_profiles['role_id'] == current_id]
    if len(cur) == 0: return []
    cur_salary = float(cur['avg_salary'].iloc[0])

    cands = []
    for _, row in role_profiles.iterrows():
        rid = int(row['role_id'])
        if rid == current_id: continue
        exp_diff = float(row['avg_exp']) - years_exp
        if -2 <= exp_diff <= 5 and float(row['avg_salary']) > cur_salary:
            uplift = (float(row['avg_salary']) / cur_salary - 1) * 100
            cands.append({
                'role': ROLES[rid],
                'avg_salary': round(float(row['avg_salary'])),
                'avg_exp': round(float(row['avg_exp']), 1),
                'salary_uplift_pct': round(uplift, 1),
                'postings_in_market': int(row['count']),
            })
    cands.sort(key=lambda c: -c['salary_uplift_pct'])
    return cands[:4]

# ==========================================
# 3. LIVE FEED WORKER
# ==========================================
recent_postings = deque(maxlen=200)
posting_counter = {'id': 0}
feed_lock = threading.Lock()
_rng = np.random.default_rng()

def generate_single_posting():
    role_id = int(_rng.integers(0, len(ROLES)))
    sen_id  = int(_rng.choice(5, p=[0.18, 0.40, 0.28, 0.10, 0.04]))
    city_id = int(_rng.choice(len(CITIES), p=CITY_PROBS))
    comp_id = int(_rng.choice(4, p=[0.30, 0.35, 0.25, 0.10]))
    ind_id  = int(_rng.choice(len(INDUSTRIES), p=INDUSTRY_PROBS))
    rem_id  = int(_rng.choice(3, p=[0.45, 0.35, 0.20]))
    years   = int(max(0, _rng.normal([1.5, 4.5, 9.0, 13.0, 17.0][sen_id], 2.5)))
    n_skills = int(np.clip(_rng.normal(4.5, 1.5), 2, 9))
    skill_list = list(_rng.choice(SKILLS, size=n_skills, replace=False))
    return {
        'role_id': role_id, 'seniority_id': sen_id, 'city_id': city_id,
        'company_tier_id': comp_id, 'industry_id': ind_id, 'remote_type_id': rem_id,
        'years_experience': years, 'skill_count': n_skills, 'skills': skill_list,
    }

def posting_to_vector(p):
    row = [p['role_id'], p['seniority_id'], p['city_id'], p['company_tier_id'],
           p['industry_id'], p['remote_type_id'], p['years_experience'], p['skill_count']]
    skill_set = set(p['skills'])
    for s in SKILLS:
        row.append(1.0 if s in skill_set else 0.0)
    return np.array([row], dtype=float)

def stream_worker():
    while True:
        try:
            p = generate_single_posting()
            x = posting_to_vector(p)
            pred = float(model.predict(x)[0])
            with feed_lock:
                posting_counter['id'] += 1
                tid = posting_counter['id']
            p['posting_id'] = f"J{tid:05d}"
            p['predicted_salary'] = round(pred, -2)
            p['role_name'] = ROLES[p['role_id']]
            p['seniority_name'] = SENIORITY[p['seniority_id']]
            p['city_name'] = CITIES[p['city_id']]
            p['company_tier_name'] = COMPANY_TIERS[p['company_tier_id']]
            p['remote_type_name'] = REMOTE_TYPES[p['remote_type_id']]
            p['ts'] = time.time()
            recent_postings.appendleft(p)
        except Exception as e:
            print(f"⚠️ stream error: {e}")
        time.sleep(3.0)

threading.Thread(target=stream_worker, daemon=True).start()

# ==========================================
# 4. FLASK API ENDPOINTS
# ==========================================
@app.route('/api/feed')
def api_feed():
    with feed_lock:
        snapshot = list(recent_postings)[:40]
    items = [{
        'posting_id': p['posting_id'], 'role_name': p['role_name'],
        'seniority_name': p['seniority_name'], 'city_name': p['city_name'],
        'company_tier_name': p['company_tier_name'], 'remote_type_name': p['remote_type_name'],
        'years_experience': p['years_experience'], 'predicted_salary': int(p['predicted_salary']),
        'ts': p['ts'],
    } for p in snapshot]
    return jsonify({'postings': items})

@app.route('/api/stats')
def api_stats():
    with feed_lock:
        snapshot = list(recent_postings)
    if not snapshot:
        return jsonify({'total': 0, 'avg_salary': 0, 'median_salary': 0, 'remote_pct': 0, 'top_role': '—'})
    salaries = [p['predicted_salary'] for p in snapshot]
    remote_count = sum(1 for p in snapshot if p['remote_type_name'] == 'remote')
    role_counts = {}
    for p in snapshot:
        role_counts[p['role_name']] = role_counts.get(p['role_name'], 0) + 1
    top_role = max(role_counts.items(), key=lambda kv: kv[1])[0]
    return jsonify({
        'total': len(snapshot), 'avg_salary': int(np.mean(salaries)),
        'median_salary': int(np.median(salaries)),
        'remote_pct': round(remote_count / len(snapshot) * 100, 1),
        'top_role': top_role,
    })

@app.route('/api/predict_salary', methods=['POST'])
def api_predict_salary():
    try:
        payload = request.get_json() or {}
        x, skill_set = build_ml_vector(payload)
        pred = float(model.predict(x)[0])
        ci_low  = pred - 1.96 * residual_std
        ci_high = pred + 1.96 * residual_std
        sv = shap_explainer.shap_values(x)
        if isinstance(sv, list): sv = sv[0]
        sv = np.array(sv)[0]
        contributions = sorted(zip(FEATURES, sv), key=lambda t: -abs(t[1]))[:8]
        feat_vals = x[0]
        feat_lookup = {f: float(feat_vals[i]) for i, f in enumerate(FEATURES)}
        shap_list = [{'feature': f, 'value': feat_lookup[f], 'impact': float(round(v, 2))} for f, v in contributions]
        return jsonify({
            'predicted_salary': round(pred, -2), 'range_low': round(ci_low, -2),
            'range_high': round(ci_high, -2), 'shap': shap_list,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/skill_gap', methods=['POST'])
def api_skill_gap():
    try:
        payload = request.get_json() or {}
        target_role = str(payload.get('target_role', '')).strip().lower()
        user_skills = payload.get('skills', [])
        results = compute_skill_gap(target_role, user_skills)
        return jsonify({'target_role': target_role, 'missing_skills': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/career_path')
def api_career_path():
    try:
        current_role = request.args.get('current_role', 'data_analyst').lower()
        years_exp = int(request.args.get('years_exp', 3))
        results = career_next_steps(current_role, years_exp)
        return jsonify({'current_role': current_role, 'years_exp': years_exp, 'recommendations': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights')
def api_insights():
    return jsonify({
        'r2_test': round(float(r2_te), 4), 'mae_test': round(float(mae_te)),
        'residual_std': round(float(residual_std)), 'n_postings': int(len(df)),
        'roles': ROLES, 'seniorities': SENIORITY, 'cities': CITIES,
        'company_tiers': COMPANY_TIERS, 'industries': INDUSTRIES,
        'remote_types': REMOTE_TYPES, 'skills': SKILLS,
        'role_stats': role_stats, 'skill_stats': skill_stats[:15],
        'importances': importance_list,
    })

# ==========================================
# 5. FRONTEND HTML
# ==========================================
HTML_PAGE = r'''<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>TalentScope AI — Job Market Intelligence</title>
<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500&display=swap' rel='stylesheet'>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
:root {
  --card: rgba(255,255,255,0.035); --border: rgba(255,255,255,0.08);
  --text: #f1f5f9; --dim: #94a3b8; --indigo: #6366f1; --cyan: #06b6d4;
  --green: #10b981; --red: #ef4444; --amber: #f59e0b; --violet: #8b5cf6;
}
body {
  font-family:'Inter',sans-serif; background: radial-gradient(ellipse at top left,#0f172a 0%,#020617 55%);
  min-height:100vh; color:var(--text); padding:14px 20px; overflow-x:hidden;
}
body::before, body::after {
  content:''; position:fixed; width:600px; height:600px; border-radius:50%;
  pointer-events:none; z-index:0; animation:float 26s ease-in-out infinite;
}
body::before { top:-230px; left:-230px; background: radial-gradient(circle, rgba(99,102,241,0.14), transparent 70%); }
body::after  { bottom:-230px; right:-230px; background: radial-gradient(circle, rgba(6,182,212,0.10), transparent 70%); animation-direction: reverse; }
@keyframes float { 0%,100%{transform:translate(0,0);} 50%{transform:translate(60px,50px);} }
.container { max-width:1640px; margin:0 auto; position:relative; z-index:1; }
header { margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }
header h1 { font-size:30px; font-weight:900; letter-spacing:-1.2px; line-height:1.05; background: linear-gradient(135deg,#a5b4fc 0%,#67e8f9 50%,#c4b5fd 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
header p { color:var(--dim); font-size:11.5px; margin-top:3px; }
.badge-live { display:inline-flex; align-items:center; gap:5px; padding:4px 11px; background: rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); color:#6ee7b7; border-radius:20px; font-size:10px; font-weight:700; letter-spacing:1px; }
.badge-live::before { content:''; width:6px; height:6px; border-radius:50%; background:#10b981; animation:pulse 1.6s infinite; }
@keyframes pulse { 0%,100%{opacity:1; box-shadow:0 0 0 0 rgba(16,185,129,0.6);} 50%{opacity:0.4; box-shadow:0 0 0 6px rgba(16,185,129,0);} }
.stats-row { display:grid; grid-template-columns: repeat(5,1fr); gap:10px; margin-bottom:12px; }
@media (max-width:1000px){ .stats-row{grid-template-columns: repeat(2,1fr);} }
.stat-card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 14px; backdrop-filter:blur(20px); }
.stat-card .label { font-size:9.5px; font-weight:700; color:var(--dim); text-transform:uppercase; letter-spacing:1px; }
.stat-card .value { font-size:19px; font-weight:800; margin-top:3px; font-family:'JetBrains Mono',monospace; letter-spacing:-0.4px; }
.stat-card .sub { font-size:10px; color:var(--dim); margin-top:2px; }
.stat-card.indigo .value { color:#a5b4fc; } .stat-card.green .value { color:#6ee7b7; } .stat-card.amber .value { color:#fcd34d; } .stat-card.violet .value { color:#c4b5fd; } .stat-card.cyan .value { color:#67e8f9; }
.main-grid { display:grid; grid-template-columns: 1.35fr 1fr; gap:12px; margin-bottom:12px; }
.main-grid + .main-grid { grid-template-columns: 1fr 1fr; }
@media (max-width:1000px){ .main-grid{grid-template-columns:1fr !important;} }
.panel { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px 16px; backdrop-filter:blur(20px); box-shadow:0 8px 40px rgba(0,0,0,0.25); }
.panel h2 { font-size:13px; font-weight:700; margin-bottom:2px; display:flex; align-items:center; gap:8px; }
.panel h2::before { content:''; width:3px; height:13px; background:linear-gradient(180deg,var(--indigo),var(--cyan)); border-radius:2px; }
.panel .subtitle { color:var(--dim); font-size:10px; margin-bottom:10px; margin-left:11px; }
.form-grid { display:grid; grid-template-columns: 1fr 1fr; gap:7px; margin-bottom:8px; }
.form-grid.three { grid-template-columns: 1fr 1fr 1fr; }
.field { margin-bottom:0; }
.field label { display:block; font-size:9.5px; font-weight:600; color:var(--dim); margin-bottom:3px; text-transform:uppercase; letter-spacing:0.6px; }
.field select, .field input { width:100%; padding:6px 9px; border-radius:7px; border:1px solid var(--border); background:rgba(0,0,0,0.35); color:var(--text); font-family:inherit; font-size:12px; outline:none; transition:all 0.2s; }
.field select:focus, .field input:focus { border-color:var(--indigo); box-shadow:0 0 0 3px rgba(99,102,241,0.1); }
.field select option { background:#0f172a; }
.skills-grid { display:grid; grid-template-columns: repeat(5,1fr); gap:4px; margin-top:5px; }
@media (max-width:700px){ .skills-grid{grid-template-columns: repeat(3,1fr);} }
.skill-chip { display:flex; align-items:center; gap:4px; padding:4px 6px; background:rgba(0,0,0,0.25); border:1px solid var(--border); border-radius:6px; font-size:10px; cursor:pointer; user-select:none; transition:all 0.15s; font-family:'JetBrains Mono',monospace; }
.skill-chip input { margin:0; accent-color: var(--indigo); cursor:pointer; width:11px; height:11px; }
.skill-chip:hover { border-color: rgba(99,102,241,0.4); }
.skill-chip.checked { background:rgba(99,102,241,0.15); border-color:rgba(99,102,241,0.5); color:#c7d2fe; }
.btn { background: linear-gradient(135deg,var(--indigo),var(--cyan)); border:none; color:white; padding:9px 16px; border-radius:8px; font-family:inherit; font-size:12px; font-weight:700; cursor:pointer; transition:all 0.2s; letter-spacing:0.3px; box-shadow:0 4px 18px rgba(99,102,241,0.3); }
.btn:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 6px 24px rgba(99,102,241,0.5); }
.btn:disabled { opacity:0.6; cursor:not-allowed; }
.btn.full { width:100%; margin-top:8px; }
.result-hero { background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(6,182,212,0.08)); border:1px solid rgba(99,102,241,0.25); border-radius:11px; padding:12px 16px; text-align:center; margin-top:8px; display:none; }
.result-hero.show { display:block; animation: fadeUp 0.4s ease; }
@keyframes fadeUp { from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:translateY(0);} }
.result-hero .label { font-size:9.5px; font-weight:700; color:var(--dim); text-transform:uppercase; letter-spacing:1.4px; }
.result-hero .price { font-size:26px; font-weight:900; letter-spacing:-1.2px; background: linear-gradient(135deg,#a5b4fc,#67e8f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:4px 0; }
.result-hero .range { color:var(--dim); font-size:10.5px; font-family:'JetBrains Mono',monospace; }
.feed { max-height: 340px; overflow-y:auto; padding-right:4px; }
.feed::-webkit-scrollbar { width:6px; }
.feed::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius:3px; }
.feed-row { display:grid; grid-template-columns: 56px 1fr 58px 72px; gap:6px; align-items:center; padding:6px 9px; border-radius:8px; margin-bottom:3px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); cursor:pointer; transition:all 0.15s; font-family:'JetBrains Mono',monospace; font-size:10.5px; animation: slideIn 0.35s ease; }
@keyframes slideIn { from{opacity:0; transform:translateX(-10px);} to{opacity:1; transform:translateX(0);} }
.feed-row:hover { background:rgba(99,102,241,0.08); border-color:rgba(99,102,241,0.25); }
.feed-row .tid { color:var(--dim); font-size:9.5px; }
.feed-row .role { color:var(--text); font-weight:600; font-family:'Inter',sans-serif; font-size:10.5px; }
.feed-row .sal { text-align:right; font-weight:700; color:#6ee7b7; }
.feed-row .city { color:var(--dim); font-size:9.5px; text-align:right; }
.gap-table { width:100%; border-collapse:collapse; font-size:11px; margin-top:4px; }
.gap-table th { text-align:left; padding:6px 5px; color:var(--dim); font-weight:600; font-size:9px; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid var(--border); }
.gap-table td { padding:6px 5px; border-bottom:1px solid rgba(255,255,255,0.04); font-family:'JetBrains Mono',monospace; font-size:11px; }
.gap-table tr:hover td { background:rgba(99,102,241,0.04); }
.gap-table .skill-name { color:#c7d2fe; font-weight:600; }
.gap-table .uplift { color:#6ee7b7; font-weight:700; }
.career-grid { display:grid; grid-template-columns: repeat(2,1fr); gap:9px; margin-top:5px; }
@media (max-width:700px){ .career-grid{grid-template-columns:1fr;} }
.career-card { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:10px; padding:11px 13px; }
.career-card .role { font-size:12.5px; font-weight:700; color:#c7d2fe; text-transform:capitalize; }
.career-card .salary { font-size:16px; font-weight:800; color:#6ee7b7; font-family:'JetBrains Mono',monospace; margin-top:4px; }
.career-card .meta { display:flex; gap:10px; margin-top:5px; font-size:10px; color:var(--dim); }
.career-card .uplift-badge { display:inline-block; padding:2px 8px; border-radius:10px; background:rgba(16,185,129,0.15); color:#6ee7b7; border:1px solid rgba(16,185,129,0.3); font-weight:700; font-size:10px; margin-left:5px; }
.charts-row { display:grid; grid-template-columns: repeat(3,1fr); gap:12px; margin-top:12px; }
@media (max-width:1000px){ .charts-row{grid-template-columns:1fr;} }
canvas { max-height:200px; }
#shap-chart { max-height:170px; }
footer { text-align:center; margin-top:14px; color:var(--dim); font-size:10px; }
.spinner { display:inline-block; width:12px; height:12px; border:2px solid rgba(255,255,255,0.3); border-top-color:white; border-radius:50%; animation:spin 0.6s linear infinite; vertical-align:middle; margin-right:6px; }
@keyframes spin { to { transform:rotate(360deg); } }
.empty-state { text-align:center; padding:26px 16px; color:var(--dim); font-size:11px; border:2px dashed rgba(255,255,255,0.06); border-radius:10px; }
</style>
</head>
<body>
<div class='container'>
  <header>
    <div>
      <h1>TalentScope AI</h1>
      <p>Job Market Intelligence · Salary Prediction · Skill Gap Analysis · Career Paths</p>
    </div>
    <span class='badge-live'>LIVE MARKET</span>
  </header>

  <div class='stats-row'>
    <div class='stat-card indigo'><div class='label'>Postings in Buffer</div><div class='value' id='s-total'>0</div><div class='sub'>live market feed</div></div>
    <div class='stat-card green'><div class='label'>Avg Salary</div><div class='value' id='s-avg'>$0</div><div class='sub'>across live postings</div></div>
    <div class='stat-card cyan'><div class='label'>Median Salary</div><div class='value' id='s-med'>$0</div><div class='sub'>50th percentile</div></div>
    <div class='stat-card amber'><div class='label'>Remote %</div><div class='value' id='s-remote'>0%</div><div class='sub'>fully-remote postings</div></div>
    <div class='stat-card violet'><div class='label'>Hottest Role</div><div class='value' id='s-role' style='font-size:14px;'>—</div><div class='sub'>most frequent in feed</div></div>
  </div>

  <div class='main-grid'>
    <div class='panel'>
      <h2>Salary Predictor</h2>
      <p class='subtitle'>Predict fair market salary + see what drives it (SHAP)</p>
      <div class='form-grid three'>
        <div class='field'><label>Role</label><select id='p-role'></select></div>
        <div class='field'><label>Seniority</label><select id='p-seniority'></select></div>
        <div class='field'><label>City</label><select id='p-city'></select></div>
      </div>
      <div class='form-grid three'>
        <div class='field'><label>Company Tier</label><select id='p-company'></select></div>
        <div class='field'><label>Industry</label><select id='p-industry'></select></div>
        <div class='field'><label>Remote</label><select id='p-remote'></select></div>
      </div>
      <div class='form-grid'>
        <div class='field'><label>Years of Experience</label><input type='number' id='p-years' value='4' min='0' max='30'></div>
        <div class='field'><label>&nbsp;</label><button class='btn full' id='predict-btn' onclick='predictSalary()' style='margin-top:0;'>Predict Salary</button></div>
      </div>
      <div style='margin-top:8px;'>
        <label style='font-size:9.5px; font-weight:600; color:var(--dim); text-transform:uppercase; letter-spacing:0.6px;'>Skills</label>
        <div class='skills-grid' id='skills-grid'></div>
      </div>
      <div class='result-hero' id='predict-result'>
        <div class='label'>Predicted Market Salary</div>
        <div class='price' id='pred-salary'>$—</div>
        <div class='range' id='pred-range'>—</div>
      </div>
      <div id='shap-container' style='margin-top:10px; display:none;'>
        <h3 style='font-size:10.5px; color:var(--dim); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>SHAP — What Drives This Prediction</h3>
        <canvas id='shap-chart' height='200'></canvas>
      </div>
    </div>

    <div class='panel'>
      <h2>Live Job Feed</h2>
      <p class='subtitle'>New postings every ~3s · click a row to load into predictor</p>
      <div class='feed' id='feed'></div>
    </div>
  </div>

  <div class='main-grid'>
    <div class='panel'>
      <h2>Skill Gap Analyzer</h2>
      <p class='subtitle'>What skills should you learn next to reach your target role?</p>
      <div class='form-grid'>
        <div class='field'><label>Target Role</label><select id='gap-role'></select></div>
        <div class='field'><label>&nbsp;</label><button class='btn full' onclick='analyzeGap()' style='margin-top:0;' id='gap-btn'>Analyze Gap</button></div>
      </div>
      <div style='margin-top:8px;'>
        <label style='font-size:9.5px; font-weight:600; color:var(--dim); text-transform:uppercase; letter-spacing:0.6px;'>Your Current Skills</label>
        <div class='skills-grid' id='gap-skills-grid'></div>
      </div>
      <div id='gap-result' style='margin-top:10px;'><div class='empty-state'>Pick a target role + your current skills, then analyze</div></div>
    </div>

    <div class='panel'>
      <h2>Career Path Explorer</h2>
      <p class='subtitle'>Where can you go next, and how much more could you earn?</p>
      <div class='form-grid'>
        <div class='field'><label>Current Role</label><select id='cp-role'></select></div>
        <div class='field'><label>Years of Experience</label><input type='number' id='cp-years' value='4' min='0' max='30'></div>
      </div>
      <button class='btn full' onclick='exploreCareer()' id='cp-btn'>Explore Next Steps</button>
      <div id='cp-result' style='margin-top:10px;'><div class='empty-state'>Enter your current role + experience to see next steps</div></div>
    </div>
  </div>

  <div class='charts-row'>
    <div class='panel'><h2>Salary by Role</h2><p class='subtitle'>Average market salary per role</p><canvas id='role-chart'></canvas></div>
    <div class='panel'><h2>Top Paying Skills</h2><p class='subtitle'>Salary uplift vs market ($)</p><canvas id='skill-chart'></canvas></div>
    <div class='panel'><h2>Feature Importance</h2><p class='subtitle'>Top 12 model drivers</p><canvas id='imp-chart'></canvas></div>
  </div>

  <footer>TalentScope AI · Gradient Boosting salary model · SHAP explanations · served via Render</footer>
</div>

<script>
var insights = null;
var selectedSkills = new Set();
var selectedGapSkills = new Set();
var shapChart = null, roleChart = null, skillChart = null, impChart = null;

function fmt(n){ return '$' + Number(n).toLocaleString(); }
function cap(s){ return s.replace(/_/g,' ').replace(/\\b\\w/g, function(c){ return c.toUpperCase(); }); }

async function loadInsights() {
  const r = await fetch('/api/insights');
  insights = await r.json();
  populateDropdowns();
  buildSkillsGrid();
  renderCharts();
}

function populateDropdowns() {
  const roleOpts = insights.roles.map(function(r,i){ return `<option value='${i}'>${cap(r)}</option>`; }).join('');
  const senOpts  = insights.seniorities.map(function(r,i){ return `<option value='${i}'>${cap(r)}</option>`; }).join('');
  const cityOpts = insights.cities.map(function(r,i){ return `<option value='${i}'>${cap(r)}</option>`; }).join('');
  const compOpts = insights.company_tiers.map(function(r,i){ return `<option value='${i}'>${cap(r)}</option>`; }).join('');
  const indOpts  = insights.industries.map(function(r,i){ return `<option value='${i}'>${cap(r)}</option>`; }).join('');
  const remOpts  = insights.remote_types.map(function(r,i){ return `<option value='${i}'>${cap(r)}</option>`; }).join('');

  ['p-role','gap-role','cp-role'].forEach(function(id){ document.getElementById(id).innerHTML = roleOpts; });
  document.getElementById('p-seniority').innerHTML = senOpts;
  document.getElementById('p-city').innerHTML = cityOpts;
  document.getElementById('p-company').innerHTML = compOpts;
  document.getElementById('p-industry').innerHTML = indOpts;
  document.getElementById('p-remote').innerHTML = remOpts;

  document.getElementById('p-role').value = 2;
  document.getElementById('p-seniority').value = 2;
  document.getElementById('p-city').value = 0;
  document.getElementById('p-company').value = 2;
  document.getElementById('p-industry').value = 4;
  document.getElementById('p-remote').value = 1;
  document.getElementById('gap-role').value = 2;
  document.getElementById('cp-role').value = 0;
}

function buildSkillsGrid() {
  const html = insights.skills.map(function(s){
    return `<label class='skill-chip' data-skill='${s}'><input type='checkbox' value='${s}' onchange="toggleSkill(this,'selectedSkills')"><span>${s}</span></label>`;
  }).join('');
  document.getElementById('skills-grid').innerHTML = html;

  const gapHtml = insights.skills.map(function(s){
    return `<label class='skill-chip' data-skill='${s}'><input type='checkbox' value='${s}' onchange="toggleSkill(this,'selectedGapSkills')"><span>${s}</span></label>`;
  }).join('');
  document.getElementById('gap-skills-grid').innerHTML = gapHtml;
}

function toggleSkill(el, setName) {
  const chip = el.closest('.skill-chip');
  const set = setName === 'selectedSkills' ? selectedSkills : selectedGapSkills;
  if (el.checked) { set.add(el.value); chip.classList.add('checked'); }
  else { set.delete(el.value); chip.classList.remove('checked'); }
}

async function predictSalary() {
  const btn = document.getElementById('predict-btn');
  btn.disabled = true; btn.innerHTML = `<span class='spinner'></span>Predicting...`;

  const payload = {
    role_id: +document.getElementById('p-role').value,
    seniority_id: +document.getElementById('p-seniority').value,
    city_id: +document.getElementById('p-city').value,
    company_tier_id: +document.getElementById('p-company').value,
    industry_id: +document.getElementById('p-industry').value,
    remote_type_id: +document.getElementById('p-remote').value,
    years_experience: +document.getElementById('p-years').value,
    skills: Array.from(selectedSkills),
  };

  try {
    const r = await fetch('/api/predict_salary', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const d = await r.json();
    if (d.error) { alert('Error: ' + d.error); return; }
    document.getElementById('pred-salary').textContent = fmt(d.predicted_salary);
    document.getElementById('pred-range').textContent = fmt(d.range_low) + '  —  ' + fmt(d.range_high) + '  (95% CI)';
    document.getElementById('predict-result').classList.add('show');
    document.getElementById('shap-container').style.display = 'block';
    renderShap(d.shap);
  } catch(e) { alert('Error: ' + e.message); }
  finally { btn.disabled = false; btn.innerHTML = 'Predict Salary'; }
}

function renderShap(shapData) {
  const labels = shapData.map(function(s){ return s.feature; });
  const values = shapData.map(function(s){ return s.impact; });
  const colors = values.map(function(v){ return v > 0 ? 'rgba(16,185,129,0.75)' : 'rgba(239,68,68,0.75)'; });

  if (shapChart) shapChart.destroy();
  const ctx = document.getElementById('shap-chart').getContext('2d');
  shapChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }] },
    options: {
      indexAxis: 'y', responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(c){ return (c.raw >= 0 ? '+' : '') + '$' + Math.round(c.raw).toLocaleString(); } } } },
      scales: {
        x: { ticks: { color:'#94a3b8', callback: function(v){ return (v>=0?'+':'') + '$' + Math.round(v/1000) + 'k'; }, font:{size:9} }, grid:{color:'rgba(255,255,255,0.05)'} },
        y: { ticks: { color:'#cbd5e1', font:{size:10} }, grid:{display:false} }
      }
    }
  });
}

async function refreshFeed() {
  try {
    const r = await fetch('/api/feed');
    const d = await r.json();
    const el = document.getElementById('feed');
    el.innerHTML = d.postings.map(function(p){
      return `<div class='feed-row' onclick="loadFromFeed(${JSON.stringify(p).replace(/"/g,'&quot;')})">
        <span class='tid'>${p.posting_id}</span>
        <span class='role'>${cap(p.role_name)} · ${p.seniority_name}</span>
        <span class='city'>${cap(p.city_name).slice(0,7)}</span>
        <span class='sal'>${fmt(p.predicted_salary)}</span>
      </div>`;
    }).join('');
  } catch(e) { console.error(e); }
}

async function refreshStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('s-total').textContent = d.total;
    document.getElementById('s-avg').textContent = fmt(d.avg_salary);
    document.getElementById('s-med').textContent = fmt(d.median_salary);
    document.getElementById('s-remote').textContent = d.remote_pct + '%';
    document.getElementById('s-role').textContent = cap(d.top_role);
  } catch(e) { console.error(e); }
}

function loadFromFeed(p) {
  document.getElementById('p-role').value = insights.roles.indexOf(p.role_name);
  document.getElementById('p-seniority').value = insights.seniorities.indexOf(p.seniority_name);
  document.getElementById('p-city').value = insights.cities.indexOf(p.city_name);
  document.getElementById('p-company').value = insights.company_tiers.indexOf(p.company_tier_name);
  document.getElementById('p-remote').value = insights.remote_types.indexOf(p.remote_type_name);
  document.getElementById('p-years').value = p.years_experience;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function analyzeGap() {
  const btn = document.getElementById('gap-btn');
  btn.disabled = true; btn.innerHTML = `<span class='spinner'></span>Analyzing...`;
  try {
    const payload = { target_role: insights.roles[+document.getElementById('gap-role').value], skills: Array.from(selectedGapSkills) };
    const r = await fetch('/api/skill_gap', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const d = await r.json();
    renderGap(d);
  } catch(e) { alert('Error: ' + e.message); }
  finally { btn.disabled = false; btn.innerHTML = 'Analyze Gap'; }
}

function renderGap(d) {
  const el = document.getElementById('gap-result');
  if (!d.missing_skills || d.missing_skills.length === 0) {
    el.innerHTML = `<div class='empty-state'>No significant gaps found — you may already be a strong fit for <strong>${cap(d.target_role)}</strong>!</div>`;
    return;
  }
  let html = `<table class='gap-table'><thead><tr><th>Skill</th><th>Required in</th><th>Avg Salary Uplift</th></tr></thead><tbody>`;
  d.missing_skills.forEach(function(s){
    html += `<tr><td class='skill-name'>${s.skill}</td><td>${s.pct_required}%</td><td class='uplift'>+$${s.salary_uplift.toLocaleString()}</td></tr>`;
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

async function exploreCareer() {
  const btn = document.getElementById('cp-btn');
  btn.disabled = true; btn.innerHTML = `<span class='spinner'></span>Exploring...`;
  try {
    const roleName = insights.roles[+document.getElementById('cp-role').value];
    const years = +document.getElementById('cp-years').value;
    const r = await fetch('/api/career_path?current_role=' + roleName + '&years_exp=' + years);
    const d = await r.json();
    renderCareer(d);
  } catch(e) { alert('Error: ' + e.message); }
  finally { btn.disabled = false; btn.innerHTML = 'Explore Next Steps'; }
}

function renderCareer(d) {
  const el = document.getElementById('cp-result');
  if (!d.recommendations || d.recommendations.length === 0) {
    el.innerHTML = `<div class='empty-state'>No clear next-step roles found for ${cap(d.current_role)} at ${d.years_exp} yrs experience.</div>`;
    return;
  }
  let html = `<div class='career-grid'>`;
  d.recommendations.forEach(function(c){
    html += `<div class='career-card'>
      <div class='role'>${cap(c.role)}<span class='uplift-badge'>+${c.salary_uplift_pct}%</span></div>
      <div class='salary'>${fmt(c.avg_salary)}</div>
      <div class='meta'><span>Avg exp: ${c.avg_exp} yrs</span><span>${c.postings_in_market} postings</span></div>
    </div>`;
  });
  html += '</div>';
  el.innerHTML = html;
}

function renderCharts() {
  const rLabels = insights.role_stats.map(function(r){ return cap(r.role); });
  const rValues = insights.role_stats.map(function(r){ return r.avg_salary; });
  if (roleChart) roleChart.destroy();
  roleChart = new Chart(document.getElementById('role-chart'), {
    type: 'bar',
    data: { labels: rLabels, datasets: [{ data: rValues, backgroundColor: 'rgba(99,102,241,0.7)', borderRadius: 4 }] },
    options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}, tooltip:{callbacks:{label:function(c){ return '$'+c.raw.toLocaleString(); }}}},
      scales: { x:{ ticks:{color:'#94a3b8', callback:function(v){return '$'+Math.round(v/1000)+'k';}, font:{size:9}}, grid:{color:'rgba(255,255,255,0.05)'} }, y:{ ticks:{color:'#cbd5e1', font:{size:10}}, grid:{display:false} } }
    }
  });

  const sLabels = insights.skill_stats.slice(0,12).map(function(s){ return s.skill; });
  const sValues = insights.skill_stats.slice(0,12).map(function(s){ return s.salary_uplift; });
  if (skillChart) skillChart.destroy();
  skillChart = new Chart(document.getElementById('skill-chart'), {
    type: 'bar',
    data: { labels: sLabels, datasets: [{ data: sValues, backgroundColor: 'rgba(16,185,129,0.7)', borderRadius: 4 }] },
    options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}, tooltip:{callbacks:{label:function(c){ return '+$'+Math.round(c.raw).toLocaleString(); }}}},
      scales: { x:{ ticks:{color:'#94a3b8', callback:function(v){return '$'+Math.round(v/1000)+'k';}, font:{size:9}}, grid:{color:'rgba(255,255,255,0.05)'} }, y:{ ticks:{color:'#cbd5e1', font:{size:10}}, grid:{display:false} } }
    }
  });

  const iLabels = insights.importances.slice(0,12).map(function(i){ return i.feature; });
  const iValues = insights.importances.slice(0,12).map(function(i){ return i.importance; });
  if (impChart) impChart.destroy();
  impChart = new Chart(document.getElementById('imp-chart'), {
    type: 'bar',
    data: { labels: iLabels, datasets: [{ data: iValues, backgroundColor: 'rgba(139,92,246,0.7)', borderRadius: 4 }] },
    options: { indexAxis:'y', responsive:true, plugins:{legend:{display:false}},
      scales: { x:{ ticks:{color:'#94a3b8', font:{size:9}}, grid:{color:'rgba(255,255,255,0.05)'} }, y:{ ticks:{color:'#cbd5e1', font:{size:10}}, grid:{display:false} } }
    }
  });
}

(async function(){
  await loadInsights();
  await refreshFeed();
  await refreshStats();
  setInterval(refreshFeed, 3000);
  setInterval(refreshStats, 3000);
})();
</script>
</body>
</html>'''

@app.route('/')
def home():
    return HTML_PAGE

# ==========================================
# 6. RUN
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
