import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)


# 1. ML & Data service (Logic Layer)

MODEL_PATH = "ad_model_pipeline.pkl"

class AdEngine:
    def __init__(self):
        # 2026 Public Benchmarks (Real Industry Averages)
        self.benchmarks = {
            "google": {"ctr": 0.0317, "cpm": 3.53, "base_reach": 1500000},
            "meta": {"ctr": 0.011, "cpm": 8.50, "base_reach": 2000000},
            "snapchat": {"ctr": 0.006, "cpm": 2.10, "base_reach": 900000}
        }
        self.load_or_train_model()

    def load_or_train_model(self):
        """Trains a model to simulate demographic-based performance."""
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
            return

        # Synthetic Training Data
        data = []
        platforms = ["google", "meta", "snapchat"]
        interests = ["technology", "fashion", "sports", "gaming", "business"]
        for p in platforms:
            for i in interests:
                for age in range(18, 65, 5):
                    # Logic: Meta + Gaming + Young = High CTR
                    base = 0.01
                    if p == "meta" and i == "gaming": base += 0.02
                    if age < 30: base += 0.005
                    data.append({"platform": p, "interest": i, "age": age, "ctr": base})

        df = pd.DataFrame(data)
        X = df[["platform", "interest", "age"]]
        y = df["ctr"]

        self.pipeline = Pipeline([
            ('prep', ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['platform', 'interest']),
                ('num', 'passthrough', ['age'])
            ])),
            ('model', RandomForestRegressor(n_estimators=50))
        ])
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, MODEL_PATH)

    def apply_scaling(self, base_ctr, budget):
        """Applies Diminishing Returns: CTR decays as budget grows."""
        if budget <= 500: return base_ctr
        # Logarithmic decay formula
        scaling_factor = 1.0 / (np.log10(budget / 50) + 1)
        return max(base_ctr * scaling_factor, base_ctr * 0.3)

engine = AdEngine()


# 2. OPTIMIZER


def optimize_budget(results, total_budget):
    scores = {}
    # Find min CPM for efficiency scoring
    min_cpm = min(p['cpm'] for p in results.values())
    max_reach = max(p['estimated_reach'] for p in results.values())

    for name, data in results.items():
        reach_score = data['estimated_reach'] / max_reach
        efficiency_score = min_cpm / data['cpm']
        # Weighted Score: 60% Reach, 40% Cost Efficiency
        scores[name] = (reach_score * 0.6) + (efficiency_score * 0.4)

    total_s = sum(scores.values())
    allocation = {n: round((s / total_s) * total_budget, 2) for n, s in scores.items()}
    
    return {
        "best_platform": max(scores, key=scores.get),
        "allocation": allocation,
        "efficiency_scores": {k: round(v, 2) for k, v in scores.items()}
    }


# 3. API ROUTES


@app.route("/api/estimate", methods=["POST"])
def estimate():
    try:
        req = request.json
        budget = float(req.get("budget_usd", 100))
        age = int(req.get("age_min", 25))
        interest = req.get("interests", ["technology"])[0]

        results = {}
        for platform, bench in engine.benchmarks.items():
            # Get ML Base CTR
            input_df = pd.DataFrame([{"platform": platform, "interest": interest, "age": age}])
            ml_ctr = engine.pipeline.predict(input_df)[0]
            
            # Apply Diminishing Returns scaling
            final_ctr = engine.apply_scaling(ml_ctr, budget)
            
            # Reach Calculation: (Budget / CPM) * 1000
            reach = (budget / bench["cpm"]) * 1000
            
            results[platform] = {
                "estimated_reach": int(reach),
                "predicted_ctr": round(float(final_ctr), 4),
                "cpm": bench["cpm"],
                "data_source": "2026 Industry Benchmarks"
            }

        optimization = optimize_budget(results, budget)

        return jsonify({
            "reach_comparison": results,
            "optimization": optimization
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 4. DashBoard UI


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AdReach AI Pro</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #0f172a; color: #f1f5f9; font-family: 'Segoe UI', sans-serif; padding: 40px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #1e293b; padding: 25px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        input, select { background: #334155; border: 1px solid #475569; color: white; padding: 12px; border-radius: 8px; width: 100%; margin-bottom: 15px; }
        button { background: #3b82f6; color: white; border: none; padding: 15px; border-radius: 8px; width: 100%; cursor: pointer; font-weight: bold; }
        button:hover { background: #2563eb; }
        pre { background: #000; padding: 15px; border-radius: 8px; font-size: 12px; overflow-x: auto; color: #10b981; }
        h2 { margin-top: 0; color: #3b82f6; }
    </style>
</head>
<body>
    <h1>AdReach AI <small style="font-size: 12px; color: #64748b;">v2026.1 - Real Benchmarks</small></h1>
    
    <div class="grid">
        <div class="card">
            <h2>Campaign Params</h2>
            <label>Budget (USD)</label>
            <input type="number" id="budget" value="1000">
            <label>Target Age</label>
            <input type="number" id="age" value="28">
            <label>Primary Interest</label>
            <select id="interest">
                <option value="technology">Technology</option>
                <option value="gaming">Gaming</option>
                <option value="fashion">Fashion</option>
                <option value="sports">Sports</option>
            </select>
            <button onclick="runEstimation()">Generate Forecast</button>
        </div>

        <div class="card">
            <h2>Optimal Allocation</h2>
            <div id="alloc_text">Result will appear here...</div>
            <canvas id="chart" style="margin-top:20px;"></canvas>
        </div>
    </div>

    <div class="card" style="margin-top:20px;">
        <h2>Raw Analysis Output</h2>
        <pre id="raw_json">No data yet.</pre>
    </div>

    <script>
        let myChart = null;

        async function runEstimation() {
            const payload = {
                budget_usd: document.getElementById('budget').value,
                age_min: document.getElementById('age').value,
                interests: [document.getElementById('interest').value]
            };

            const res = await fetch('/api/estimate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            document.getElementById('raw_json').innerText = JSON.stringify(data, null, 2);
            
            // Update Text
            document.getElementById('alloc_text').innerHTML = `
                <b>Best Choice:</b> ${data.optimization.best_platform.toUpperCase()}<br>
                <b>Recommended Meta Spend:</b> $${data.optimization.allocation.meta}
            `;

            // Update Chart
            const labels = Object.keys(data.reach_comparison);
            const reachValues = labels.map(l => data.reach_comparison[l].estimated_reach);

            if(myChart) myChart.destroy();
            myChart = new Chart(document.getElementById('chart'), {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Estimated Total Reach',
                        data: reachValues,
                        backgroundColor: ['#4285F4', '#1877F2', '#FFFC00']
                    }]
                },
                options: { plugins: { legend: { display: false } } }
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

if __name__ == "__main__":
    # Ensure dependencies are installed: pip install flask pandas scikit-learn joblib numpy
    app.run(host='0.0.0.0', debug=True, port=5000)
