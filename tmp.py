import pandas as pd

def read_csvs_():
    df1 = pd.read_csv("./data_dir/prosmith_test4/train_val/val.csv")
    print(df1)
    print(df1.isnull().sum())
    return


def main():
    read_csvs_()
    return


if __name__ == "__main__":
    main()
