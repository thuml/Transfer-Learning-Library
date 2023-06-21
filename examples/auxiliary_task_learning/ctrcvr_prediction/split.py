import pandas as pd
import os


def split(input_file_names, output_file_names):
    print("split {} into {}".format(input_file_names, output_file_names))
    os.makedirs(os.path.dirname(output_file_names), exist_ok=True)
    data = pd.read_csv(input_file_names)
    print(data.shape)
    start_line = 0
    count = 0
    while start_line < data.shape[0]:
        end_line = min(start_line + 204800, data.shape[0])
        save_data = data.iloc[start_line:end_line]
        save_data.to_csv(output_file_names + str(count) + ".csv", index=False)
        start_line = end_line
        count += 1


if __name__ == '__main__':
    for dataset_name in ["ES", "FR", "NL", "US"]:
        for part in ["train", "test"]:
            split("data/aliexpress/AliExpress_{}/{}.csv".format(dataset_name, part),
                  "data/{}/{}_".format(dataset_name, part))
