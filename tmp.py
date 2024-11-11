import pandas as pd

def read_csvs_():
    df1 = pd.read_csv("./data_dir/prosmith_test4/train_val/train.csv")
    print(df1)
    binding = df1['output'].sum()
    non_binding = df1['output'].count() - binding
    total = df1['output'].count()
    print(f"Binding:     {binding}")
    print(f"Non-binding: {non_binding}")
    print(f"Total:       {total}")
    return


def main():
    read_csvs_()
    return


if __name__ == "__main__":
    main()
