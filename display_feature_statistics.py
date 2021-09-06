import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_pickle("transformed_data.pkl")

    gender_counted = data["Gender"].value_counts()
    x = ["Male", "Female"]
    y = [gender_counted[0], gender_counted[1]]
    plt.bar(x, y)
    plt.show()
    print("These are the relative frequencies of male and female customers: " + "Male: " + str(gender_counted[0]/data.shape[0]) + " Female: " + str(gender_counted[1]/data.shape[0]))

    plt.boxplot(data["Age"], labels = ["Age"])
    plt.show()
    plt.boxplot(data["Annual Income (k$)"], labels = ["Annual Income (k$)"])
    plt.show()
    plt.boxplot(data["Spending Score (1-100)"], labels = ["Spending Score (1-100)"])
    plt.show()


if __name__ == "__main__":
    main()
