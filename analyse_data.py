import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("raw_data/mall_customers.csv", decimal = ",")
    print(data.describe())

    print(data.corr(method = "pearson", min_periods = 1))

    plt.figure(figsize=(9, 6))
    hm = sb.heatmap(data.corr(method = "pearson", min_periods = 1), annot = True)
    hm.set_yticklabels(hm.get_yticklabels(), rotation = 0, fontsize = 8)
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    plt.show()

if __name__ == "__main__":
    main()
