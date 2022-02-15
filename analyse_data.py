import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def shorten_column_names(dataframe):
    dataframe.columns = ["GEN", "Age", "INC", "Score"]
    return dataframe
    

def main():
    data = pd.read_pickle("transformed_data.pkl")
    data = shorten_column_names(data)

    print("Statistical metrics of the given data:")
    print(data.describe())

    print("\nCorrelation of the columns of the given data:")
    print(data.corr(method = "pearson", min_periods = 1))

    plt.figure(figsize=(9, 6))
    hm = sb.heatmap(data.corr(method = "pearson", min_periods = 1), annot = True)
    hm.set_yticklabels(hm.get_yticklabels(), rotation = 0, fontsize = 8)
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    plt.show()


if __name__ == "__main__":
    main()
