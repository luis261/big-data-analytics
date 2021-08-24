import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    data = pd.read_csv("raw_data/mall_customers.csv", decimal = ",")
    # map string values to floats
    data["Gender"] = data["Gender"].map({"Female": 1, "Male": 0})

    print(data.head())
    print(data.describe())

    print(data.corr(method = "pearson", min_periods = 1))

    plt.figure(figsize=(9, 6))
    hm = sb.heatmap(data.corr(method = "pearson", min_periods = 1), annot = True)
    hm.set_yticklabels(hm.get_yticklabels(), rotation = 0, fontsize = 8)
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    # TODO better labels
    plt.show()

    K = range(1, 10)
    variance = []
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        variance.append(kmeanModel.inertia_)

    plt.figure(figsize=(16,8))
    plt.plot(K, variance, "bx-")
    plt.xlabel("k")
    plt.ylabel("Variance")
    plt.title("Scree-Plot")
    plt.show()

    km = KMeans(n_clusters = 3)
    km.fit(data)
    labels = km.fit_predict(data)
    labels = pd.DataFrame(labels, columns=["cluster"])
    clustered_data = pd.concat([data, labels], axis= 1)
    print(clustered_data.head())

    cluster0 = clustered_data[clustered_data["cluster"] == 0]
    cluster1 = clustered_data[clustered_data["cluster"] == 1]
    cluster2 = clustered_data[clustered_data["cluster"] == 2]

    plt.scatter(cluster0["Annual Income (k$)"], cluster0["Spending Score (1-100)"], color = "red")
    plt.scatter(cluster1["Annual Income (k$)"], cluster1["Spending Score (1-100)"], color = "black")
    plt.scatter(cluster2["Annual Income (k$)"], cluster2["Spending Score (1-100)"], color = "blue")
    plt.show()

"""
TODO:
- maybe omit gender data, since there is no significant correlation with any other value
- how many dimensions to analyse? income, spending score and age or only income and spending score?
- silhouette score, other evaluation metrics
"""

if __name__ == "__main__":
    main()
