import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

    # TODO Visualisation: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    for k in range(2, 10):
        scores = []
        for _ in range(5):
            model = KMeans(n_clusters=k, init = "k-means++")
            labels = model.fit_predict(data)
            scores.append(silhouette_score(data, labels))
        print("For " + str(k) + " clusters, the silhouette score is: " + str(sum(scores)/len(scores)))

    # TODO 2 seperate clusterings, one per gender for better visualisation => or one plot, with different forms
    model = KMeans(n_clusters = 5)
    labels = model.fit_predict(data)
    labels = pd.DataFrame(labels, columns=["cluster"])
    clustered_data = pd.concat([data, labels], axis= 1)

    cluster0 = clustered_data[clustered_data["cluster"] == 0]
    cluster1 = clustered_data[clustered_data["cluster"] == 1]
    cluster2 = clustered_data[clustered_data["cluster"] == 2]
    cluster3 = clustered_data[clustered_data["cluster"] == 3]
    cluster4 = clustered_data[clustered_data["cluster"] == 4]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data["Age"], data["Annual Income (k$)"], data["Spending Score (1-100)"], c = "#f23c14")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(cluster0["Age"], cluster0["Annual Income (k$)"], cluster0["Spending Score (1-100)"], c = "#f23c14")
    ax.scatter(cluster1["Age"], cluster1["Annual Income (k$)"], cluster1["Spending Score (1-100)"], c = "#72f214")
    ax.scatter(cluster2["Age"], cluster2["Annual Income (k$)"], cluster2["Spending Score (1-100)"], c = "#14f2e5")
    ax.scatter(cluster3["Age"], cluster3["Annual Income (k$)"], cluster3["Spending Score (1-100)"], c = "#f214cd")
    ax.scatter(cluster4["Age"], cluster4["Annual Income (k$)"], cluster4["Spending Score (1-100)"], c = "#2c14f2")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()


if __name__ == "__main__":
    main()
