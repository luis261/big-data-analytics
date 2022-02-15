import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def main():
    file_name = "transformed_data.pkl"
    data = pd.read_csv("raw_data/mall_customers.csv", decimal = ",")

    data.drop(columns=["CustomerID"], inplace=True)

    data["Gender"] = data["Gender"].map({"Female": 1, "Male": 0})

    columns_to_scale = list(data.columns)
    columns_to_scale.pop(0)
    data[columns_to_scale] = MinMaxScaler().fit_transform(data[columns_to_scale])

    print("Transformed data:")
    print(data)

    data.to_pickle(file_name)
    print("\nPersisted the transformed data in the file \"" + file_name + "\"")


if __name__ == "__main__":
    main()
