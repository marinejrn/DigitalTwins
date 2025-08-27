import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
import pacmap
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon

def load_data_from_csv():
    df = pd.read_csv("ml-api\model\df.csv")
    return df

def project_and_cluster_with_patient(patient_dict):
    df = load_data_from_csv()

    features = ['age_at_diagnosis', 'bmi', 'comorbidities', 'hormonal', 'hemoglobin',
                'lymphocyte_count', 'lymph_nodes_involved', 'tumor_size', 'tumor_grade']

    patient_df = pd.DataFrame([patient_dict])
    all_data = pd.concat([df[features], patient_df], ignore_index=True)

    if all_data.isnull().any().any():
        raise ValueError("Des valeurs manquantes détectées dans les données combinées.")

    reducer = pacmap.PaCMAP(n_components=3, random_state=42)
    embedded = reducer.fit_transform(all_data.values)

    clusterer = MeanShift()
    clusters = clusterer.fit_predict(embedded)

    new_point_cluster = int(clusters[-1])
    df['cluster'] = clusters[:-1]

    distances = pairwise_distances(df[features], patient_df).flatten()
    df['distance'] = distances

    return {
        "cluster": new_point_cluster,
        "coordinates_3d": embedded[-1:].flatten().tolist(),
        "df_clustered": df,
        "X_3d": embedded[:-1],
        "patient_point": embedded[-1:]
    }

def estimate_survival(patient_3d, X_3d, df, k=15):
    distances = pairwise_distances(X_3d, patient_3d).flatten()
    df = df.copy()
    df["distance"] = distances
    chemo = df[df["chemotherapy"] == 1].nsmallest(k, "distance")
    no_chemo = df[df["chemotherapy"] == 0].nsmallest(k, "distance")
    return chemo["survival_5year"].mean(), no_chemo["survival_5year"].mean()

def compute_js_divergence(g1, g2, features, n_permutations=300):
    mean1 = g1[features].mean().values
    mean2 = g2[features].mean().values
    real_js = jensenshannon(mean1, mean2) ** 2

    combined = pd.concat([g1, g2])
    n1 = len(g1)
    sims = []
    for _ in range(n_permutations):
        shuffled = combined.sample(frac=1).reset_index(drop=True)
        fake1 = shuffled.iloc[:n1]
        fake2 = shuffled.iloc[n1:]
        js = jensenshannon(fake1[features].mean(), fake2[features].mean()) ** 2
        sims.append(js)

    p_val = sum(sim >= real_js for sim in sims) / n_permutations
    return round(real_js, 3), round(p_val, 3)
