import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the pickle model safely
MODEL_PATH = "logistic_pkl.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Define the 12 features in the exact order the model expects
FEATURE_KEYS = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# Clean HTML/CSS layout embedded directly in app.py for easy single-file deployment
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #1f2937;
            --text-muted: #4b5563;
            --border: #e5e7eb;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            max-width: 800px;
            width: 100%;
        }
        .card {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        }
        h1 {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #111827;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 32px;
            font-size: 15px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.2s;
            background-color: #fafafa;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        .btn-submit {
            grid-column: span 2;
            background: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 12px;
            transition: background 0.2s;
        }
        @media (max-width: 600px) { .btn-submit { grid-column: span 1; } }
        .btn-submit:hover { background: var(--primary-hover); }
        
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }
        .risk-high {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .risk-low {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="card">
        <h1>Health Risk Predictor</h1>
        <p class="subtitle">Enter patient metrics below to evaluate clinical risk classification.</p>
        
        <form method="POST" action="/">
            <div class="form-grid">
                
                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="age" step="any" required placeholder="e.g. 60">
                </div>

                <div class="form-group">
                    <label>Anaemia</label>
                    <select name="anaemia" required>
                        <option value="0">No Anaemia</option>
                        <option value="1">Has Anaemia</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Creatinine Phosphokinase (CPK)</label>
                    <input type="number" name="creatinine_phosphokinase" step="any" required placeholder="e.g. 250">
                </div>

                <div class="form-group">
                    <label>Diabetes</label>
                    <select name="diabetes" required>
                        <option value="0">No Diabetes</option>
                        <option value="1">Has Diabetes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Ejection Fraction (%)</label>
                    <input type="number" name="ejection_fraction" step="any" required placeholder="e.g. 38">
                </div>

                <div class="form-group">
                    <label>High Blood Pressure</label>
                    <select name="high_blood_pressure" required>
                        <option value="0">Normal/No</option>
                        <option value="1">Yes/Hypertension</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Platelets Count</label>
                    <input type="number" name="platelets" step="any" required placeholder="e.g. 263000">
                </div>

                <div class="form-group">
                    <label>Serum Creatinine</label>
                    <input type="number" name="serum_creatinine" step="any" required placeholder="e.g. 1.1">
                </div>

                <div class="form-group">
                    <label>Serum Sodium</label>
                    <input type="number" name="serum_sodium" step="any" required placeholder="e.g. 136">
                </div>

                <div class="form-group">
                    <label>Sex</label>
                    <select name="sex" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Smoking Status</label>
                    <select name="smoking" required>
                        <option value="0">Non-Smoker</option>
                        <option value="1">Smoker</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Follow-up Time (Days)</label>
                    <input type="number" name="time" step="any" required placeholder="e.g. 120">
                </div>

                <button type="submit" class="btn-submit">Analyze Patient Profile</button>
            </div>
        </form>

        {% if prediction_text %}
            <div class="result-box {% if 'High' in prediction_text %}risk-high{% else %}risk-low{% endif %}">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        try:
            # Parse form features in precise model order
            input_features = []
            for key in FEATURE_KEYS:
                val = float(request.form[key])
                input_features.append(val)
            
            # Reshape array for model inference
            final_features = np.array([input_features])
            
            # Predict raw binary value (0 or 1)
            raw_prediction = model.predict(final_features)[0]
            
            # Transform numeric value into a human-readable categorical string
            if int(raw_prediction) == 1:
                prediction_text = "Result: High Risk Classification Detected"
            else:
                prediction_text = "Result: Low Risk Classification Detected"
                
        except Exception as e:
            prediction_text = f"Error handling request: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
