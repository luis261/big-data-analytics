import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np


def main():
    data = pd.read_pickle("transformed_data.pkl")

    K = range(1, 10)
    inertia_per_k = []
    for k in K:
        inertia_per_run = []
        for _ in range(5):
            model = KMeans(n_clusters=k, init = "k-means++")
            model.fit(data)
            inertia_per_run.append(model.inertia_)
        inertia_per_k.append(sum(inertia_per_run)/len(inertia_per_run))

    plt.figure(figsize=(16,8))
    plt.plot(K, inertia_per_k, "bx-")
    plt.xlabel("k")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("Scree-Plot")
    plt.show()

    # The visualization of the silhouette scores was copied in parts from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    for k in range(2, 10):
        scores = []
        for _ in range(5):
            model = KMeans(n_clusters=k, init = "k-means++")
            labels = model.fit_predict(data)
            scores.append(silhouette_score(data, labels))
        print("For " + str(k) + " clusters, the silhouette score is: " + str(sum(scores)/len(scores)) + ".")

        # The (k + 1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        plt.ylim([0, len(data) + (k + 1) * 10])

        sample_silhouette_values = silhouette_samples(data, labels)
        y_lower = 10
        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        plt.show()

    print("\n")

    model = KMeans(n_clusters = 5)
    labels = model.fit_predict(data)
    labels = pd.DataFrame(labels, columns=["cluster"])
    clustered_data = pd.concat([data, labels], axis= 1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data["Age"], data["Annual Income (k$)"], data["Spending Score (1-100)"], c = "#f23c14")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()

    clusters = []
    clusters.append(clustered_data[clustered_data["cluster"] == 0])
    clusters.append(clustered_data[clustered_data["cluster"] == 1])
    clusters.append(clustered_data[clustered_data["cluster"] == 2])
    clusters.append(clustered_data[clustered_data["cluster"] == 3])
    clusters.append(clustered_data[clustered_data["cluster"] == 4])

    # separate customers by gender to be able to dispay them differently
    male_points_per_cluster = []
    female_points_per_cluster = []
    for cluster in clusters:
        male_points_per_cluster.append(cluster[cluster["Gender"] == 0])
        female_points_per_cluster.append(cluster[cluster["Gender"] == 1])
    clusters_separated_by_gender = [male_points_per_cluster, female_points_per_cluster]
    colors = ["#f23c14", "#72f214", "#14f2e5", "#f214cd", "#2c14f2"]
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
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()


if __name__ == "__main__":
    main()
