import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ColorException(Exception):
    pass

def cluster_data_and_show(data, k):
    if k > 5:
        raise ColorException("k exceeds amount of available colors")
    else:
        model = KMeans(n_clusters = k)
        labels = model.fit_predict(data)
        labels = pd.DataFrame(labels, columns=["cluster"])
        clustered_data = pd.concat([data, labels], axis= 1)
        clusters = []
        for i in range(k):
            clusters.append(clustered_data[clustered_data["cluster"] == i])

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

    averaged_scores = []
    for k in range(2, 10):
        scores = []
        for _ in range(5):
            model = KMeans(n_clusters=k, init = "k-means++")
            labels = model.fit_predict(data)
            scores.append(silhouette_score(data, labels))
        averaged_scores.append(sum(scores)/len(scores))
        print("For " + str(k) + " clusters, the silhouette score is: " + str(averaged_scores[-1]) + ".")
    print("\n")

    plt.bar(range(2, 10), averaged_scores)
    plt.show()


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data["Age"], data["Annual Income (k$)"], data["Spending Score (1-100)"], c = "#f23c14")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.show()

    cluster_data_and_show(data, 3)
    cluster_data_and_show(data, 5)


if __name__ == "__main__":
    main()
