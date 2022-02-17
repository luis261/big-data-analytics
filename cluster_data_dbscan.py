from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ColorException(Exception):
    pass


def cluster_data(data, params, n_jobs=2, label=True):
    gender_data = data["Gender"]
    data.drop(columns=["Gender"], inplace=True)
    (eps, min_samples) = params
    print("Clustering the data.")
    model = DBSCAN(min_samples=min_samples, eps=eps, n_jobs=n_jobs)
    clustering = model.fit(data)

    data_without_noise = data[clustering.labels_ != -1]
    clusters_amount = len(np.unique(clustering.labels_[clustering.labels_ != -1]))
    print("Clustered the given data, there are " + str(clusters_amount) + " clusters.")
    score = None
    if (clusters_amount > 1):
        score = silhouette_score(data_without_noise, clustering.labels_[clustering.labels_ != -1])
        score -= (len(data) - len(data_without_noise))/len(data)
        print("The adjusted silhouette score is: " + str(score) + ".")

    if label:
        data["cluster"] = clustering.labels_

    data["Gender"] = gender_data
    return (data, score)

def optimize_epsilon(min_samples, eps_choices, data):
    epsilons_to_scores = {}
    for eps in eps_choices:
        (data, score) = cluster_data(data, (eps, min_samples), label=False)
        if (score != None):
            epsilons_to_scores[eps] = score

    top_score = -1.0
    best_eps = None
    for eps, score in epsilons_to_scores.items():
        if score >= top_score:
            top_score = score
            best_eps = eps

    return best_eps

def plot_clustered_data(clustered_data):
    noise_exists = True
    clusters = []
    clusters_amount = len(np.unique(clustered_data[clustered_data["cluster"] != -1]["cluster"]))
    for i in range(-1, clusters_amount):
        cluster = clustered_data[clustered_data["cluster"] == i]
        if (i == -1 and len(cluster) == 0):
            noise_exists = False
            continue

        clusters.append(cluster)

    # separate customers by gender to be able to dispay them differently
    male_points_per_cluster = []
    female_points_per_cluster = []
    for cluster in clusters:
        male_points_per_cluster.append(cluster[cluster["Gender"] == 0])
        female_points_per_cluster.append(cluster[cluster["Gender"] == 1])
    clusters_separated_by_gender = [male_points_per_cluster, female_points_per_cluster]
    colors = ["#000000", "#f23c14", "#72f214", "#14f2e5", "#f214cd", "#2c14f2", "#5e008a"]
    if not noise_exists:
        colors.pop(0)

    if len(clusters) > len(colors):
        raise ColorException("Not enough colors!")

    markers = ["o", "x"]

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(clusters_separated_by_gender[0])):
        customers_per_gender = {}
        for j in range(len(clusters_separated_by_gender)):
            gender = "male" if j == 0 else "female"
            try:
                customers_per_gender[gender] += clusters_separated_by_gender[j][i].shape[0]
            except KeyError:
                customers_per_gender[gender] = clusters_separated_by_gender[j][i].shape[0]
            ax.scatter(clusters_separated_by_gender[j][i]["Age"], clusters_separated_by_gender[j][i]["Annual Income (k$)"], clusters_separated_by_gender[j][i]["Spending Score (1-100)"], c = colors[i], marker = markers[j])
        print("The cluster with the color " + colors[i] + " has " + str((100 * customers_per_gender["male"])/(customers_per_gender["male"] + customers_per_gender["female"])) + "% male customers.")
    print("\n")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()


def main():
    data = pd.read_pickle("transformed_data.pkl")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data["Age"], data["Annual Income (k$)"], data["Spending Score (1-100)"], c = "#f23c14")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()

    min_samples = 6
    # epsilon options start with 0.01 and go up to 1.73 in steps of 0.01
    # (3^0.5 is the biggest possible distance between 2 points in a three-dimensional cube)
    # eps = optimize_epsilon(min_samples, [x/100 for x in range(1, 173)], data)
    eps = 0.14
    (clustered_data, _) = cluster_data(data, (eps, min_samples))
    plot_clustered_data(clustered_data)


if __name__ == "__main__":
    main()
