from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cluster_data(data, params, n_jobs=2):
    (eps, min_samples) = params
    print("Clustering the data.")
    model = DBSCAN(min_samples=min_samples, eps=eps, n_jobs=n_jobs)
    clustering = model.fit(data)

    clusters_amount = len(np.unique(clustering.labels_)) - 1
    print("Clustered the given data, there are " + str(clusters_amount) + " clusters.")

    data_without_noise = data[clustering.labels_ != -1]
    score = None
    if (len(np.unique(clustering.labels_[clustering.labels_ != -1])) > 1):
        score = silhouette_score(data_without_noise, clustering.labels_[clustering.labels_ != -1])
        print("The silhouette score is: " + str(score) + ".")
        data["cluster"] = clustering.labels_

    return (data, score)

def optimize_epsilon(min_samples, eps_choices, data):
    epsilons_to_scores = {}
    for eps in eps_choices:
        (data, score) = cluster_data(data, (eps, min_samples))
        if (score != None):
            epsilons_to_scores[eps] = score

    top_score = -1.0
    best_eps = None
    for eps, score in epsilons_to_scores.items():
        if score >= top_score:
            top_score = score
            best_eps = eps

    return best_eps


def main():
    data = pd.read_pickle("transformed_data.pkl")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data["Age"], data["Annual Income (k$)"], data["Spending Score (1-100)"], c = "#f23c14")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()

    eps = optimize_epsilon(10, [x/100 for x in range(1, 101)], data)
    # TODO cluster with best eps, plot clustered data


if __name__ == "__main__":
    main()
