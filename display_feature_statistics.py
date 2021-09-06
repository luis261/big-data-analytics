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
    # TODO other features as well


if __name__ == "__main__":
    main()
