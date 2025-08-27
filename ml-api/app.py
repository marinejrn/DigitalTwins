import os
from flask import Flask, request, jsonify
from utils import project_and_cluster_with_patient,compute_js_divergence, estimate_survival
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = project_and_cluster_with_patient(data)
    
    features = ['age_at_diagnosis', 'bmi', 'comorbidities', 'hormonal', 'hemoglobin',
                'lymphocyte_count', 'lymph_nodes_involved', 'tumor_size', 'tumor_grade']
    
    surv_chemo, surv_no_chemo = estimate_survival(result["patient_point"], result["X_3d"], result["df_clustered"])

    g1 = result["df_clustered"][result["df_clustered"]["chemotherapy"] == 1].nsmallest(15, "distance")
    g2 = result["df_clustered"][result["df_clustered"]["chemotherapy"] == 0].nsmallest(15, "distance")
    js_div, p_val = compute_js_divergence(g1, g2, features)

    return jsonify({
        "cluster": result["cluster"],
        "coordinates_3d": result["patient_point"].flatten().tolist(),
        "survival_chemo": round(float(surv_chemo), 3),
        "survival_no_chemo": round(float(surv_no_chemo), 3),
        "manifold": result["X_3d"].tolist(),
        "js_divergence": js_div,
        "js_pvalue": p_val,
        "clusters": result["df_clustered"]["cluster"].tolist()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render injecte le port ici
    app.run(host="0.0.0.0", port=port, debug=False)


