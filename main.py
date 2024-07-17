import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from datetime import datetime
from models.handler import train, test, forward_test
import argparse
import pandas as pd
import numpy as np
from data_centric_approach import *

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--evaluate", type=bool, default=True)
parser.add_argument("--forward_test", type=bool, default=False)
parser.add_argument(
    "--dataset",
    type=str,
    default="EURUSD60_OHLC_MACD_MA10_MA20",
)
parser.add_argument("--window_size", type=int, default=12)
parser.add_argument("--horizon", type=int, default=3)
parser.add_argument("--train_length", type=float, default=7)
parser.add_argument("--valid_length", type=float, default=2)
parser.add_argument("--test_length", type=float, default=1)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--multi_layer", type=int, default=5)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--validate_freq", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--norm_method", type=str, default="z_score")
parser.add_argument("--optimizer", type=str, default="RMSProp")
parser.add_argument("--early_stop", type=bool, default=False)
parser.add_argument("--exponential_decay_step", type=int, default=5)
parser.add_argument("--decay_rate", type=float, default=0.5)
parser.add_argument("--dropout_rate", type=float, default=0.5)
parser.add_argument("--leakyrelu_rate", type=int, default=0.2)
parser.add_argument("--data_centric_approach", type=bool, default=False)


args = parser.parse_args()
print(f"Training configs: {args}")
data_file = os.path.join("dataset", args.dataset + ".csv")
result_train_file = os.path.join("output", args.dataset, "train")
result_test_file = os.path.join("output", args.dataset, "test")
result_forward_test_file = os.path.join("output", args.dataset, "forward_test")
data_centric_file = os.path.join(
    "output", args.dataset, "data_centric_approach.csv"
)  # Updated line for CSV file path

if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
if not os.path.exists(result_forward_test_file):
    os.makedirs(result_forward_test_file)
if not os.path.exists(
    os.path.dirname(data_centric_file)
):  # Ensure directory for data-centric file exists
    os.makedirs(os.path.dirname(data_centric_file))

data = pd.read_csv(data_file)

# data centric approach
if args.data_centric_approach:
    data = calculate_rolling_statistics(data)  # Calculate rolling statistics
    data = create_lagged_features(data)  # Create lagged features

    # Save the data after applying data-centric approach to a CSV file
    data.to_csv(data_centric_file, index=False)  # Save DataFrame to CSV

if args.forward_test:
    forward_data = data.copy()
    forward_data = extract_forward_data(forward_data)
    forward_data = forward_data.values

data = data_formating(data)  # Now, remove the header before training the model

# print("\n2", data)
data = data.values

# split data
train_ratio = args.train_length / (
    args.train_length + args.valid_length + args.test_length
)
valid_ratio = args.valid_length / (
    args.train_length + args.valid_length + args.test_length
)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[: int(train_ratio * len(data))]
valid_data = data[
    int(train_ratio * len(data)) : int((train_ratio + valid_ratio) * len(data))
]
test_data = data[int((train_ratio + valid_ratio) * len(data)) :]

torch.manual_seed(0)

if __name__ == "__main__":
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(
                train_data, valid_data, args, result_train_file
            )
            after_train = datetime.now().timestamp()
            print(f"Training took {(after_train - before_train) / 60} minutes")
        except KeyboardInterrupt:
            print("-" * 99)
            print("Exiting from training early")
    if args.evaluate:
        print("Perform test data evaluation with size: ", len(test_data))
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f"Evaluation took {(after_evaluation - before_evaluation) / 60} minutes")
    if args.forward_test:
        print("Perform out sample data evaluation with size: ", len(forward_data))
        before_evaluation = datetime.now().timestamp()
        forward_test(forward_data, args, result_train_file, result_forward_test_file)
        after_evaluation = datetime.now().timestamp()
        print(
            f"Out sample data evaluation took {(after_evaluation - before_evaluation) / 60} minutes"
        )
    print("done")
