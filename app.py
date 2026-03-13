"""
FraudGuard — Flask API + Dashboard

Run locally:
    pip install flask scikit-learn pandas numpy gunicorn
    python3 app.py
    open http://localhost:8000
"""
import os, json, math, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)

_pkg = _analysis = None

def get_model():
    global _pkg
    if _pkg is None:
        with open(os.path.join(BASE, 'models', 'best_model.pkl'), 'rb') as f:
            _pkg = pickle.load(f)
    return _pkg

def get_analysis():
    global _analysis
    if _analysis is None:
        with open(os.path.join(BASE, 'static', 'analysis.json')) as f:
            _analysis = json.load(f)
    return _analysis

def cj(obj):
    if isinstance(obj, float): return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: cj(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [cj(v) for v in obj]
    return obj

def scale_input(pkg, v_vals, amount, time):
    """Scale Amount and Time the same way as training."""
    amount_sc = (amount - pkg['amount_scaler_mean']) / (pkg['amount_scaler_std'] + 1e-8)
    time_sc   = (time   - pkg['time_scaler_mean'])   / (pkg['time_scaler_std']   + 1e-8)
    return np.array(v_vals + [amount_sc, time_sc]).reshape(1, -1)

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE, 'static'), 'index.html')

@app.route('/api/health')
def health():
    pkg = get_model(); a = get_analysis()
    return jsonify({'status': 'ok', 'model': pkg['name'], 'auc': a['meta']['best_auc']})

@app.route('/api/analysis')
def analysis():
    return Response(
        json.dumps(cj(get_analysis()), separators=(',', ':')),
        mimetype='application/json'
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    if not body:
        return jsonify({'detail': 'No JSON body'}), 400

    # Validate V1-V28
    v_vals = []
    for i in range(1, 29):
        key = f'V{i}'
        if key not in body:
            return jsonify({'detail': f'Missing field: {key}'}), 422
        v_vals.append(float(body[key]))

    for k in ['Amount', 'Time']:
        if k not in body:
            return jsonify({'detail': f'Missing field: {k}'}), 422

    amount = float(body['Amount'])
    time   = float(body['Time'])

    pkg   = get_model()
    model = pkg['model']
    row   = scale_input(pkg, v_vals, amount, time)

    prob = float(model.predict_proba(row)[0, 1])
    pred = 'FRAUD' if prob >= 0.5 else 'LEGITIMATE'
    risk = 'CRITICAL' if prob >= 0.8 else 'HIGH' if prob >= 0.5 else 'MEDIUM' if prob >= 0.2 else 'LOW'

    # Feature contributions
    a        = get_analysis()
    imp_vals = a['importance'][a['meta']['best_model']]['values']
    feat_disp = pkg['feat_display']
    raw_vals  = v_vals + [amount, time]
    raw_row   = row[0].tolist()

    contribs = sorted([{
        'feature':      feat_disp[i],
        'value':        round(raw_vals[i], 4),
        'contribution': float(raw_row[i] * imp_vals[i])
    } for i in range(len(raw_row))], key=lambda x: abs(x['contribution']), reverse=True)

    return jsonify({
        'fraud_probability': round(prob, 4),
        'prediction':  pred,
        'risk_level':  risk,
        'model_used':  pkg['name'],
        'top_features': contribs[:5],
    })

@app.route('/api/example/fraud')
def ex_fraud():
    # High-fraud-signal transaction (extreme V14, V4, etc.)
    ex = {f'V{i}': 0.0 for i in range(1, 29)}
    ex.update({'V14': -12.0, 'V4': 6.0, 'V11': 5.0, 'V3': -7.0,
               'V17': -4.0, 'V12': 5.0, 'V2': -4.0, 'V7': -3.0,
               'Amount': 149.62, 'Time': 406.0})
    return jsonify(ex)

@app.route('/api/example/legit')
def ex_legit():
    ex = {f'V{i}': round(float(np.random.RandomState(7).randn()), 3) for i in range(1, 29)}
    ex.update({'V1': -1.36, 'V2': -0.07, 'V3': 2.54, 'V4': 1.38,
               'Amount': 2.69, 'Time': 0.0})
    return jsonify(ex)

@app.route('/static/<path:fn>')
def static_files(fn):
    return send_from_directory(os.path.join(BASE, 'static'), fn)

if __name__ == '__main__':
    try:
        a = get_analysis(); pkg = get_model()
        print('=' * 50)
        print('  FraudGuard is running!')
        print(f"  Model : {pkg['name']}")
        print(f"  AUC   : {a['meta']['best_auc']:.4f}")
        print('  Open  : http://localhost:8000')
        print('=' * 50)
    except Exception as e:
        print(f'Warning: {e}')
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
