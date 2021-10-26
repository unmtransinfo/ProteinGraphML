#!/usr/bin/env python3
###
import pickle
import pyreadr
import argparse


def extract_train_predict(rdsfile):
    """
    Using the RDS file, create dataframe for training and prediction data.
    """
    print("Extract train and predict data from RDS file")
    rdsdata = pyreadr.read_r(rdsfile)
    df = rdsdata[None]
    df = df.rename(
        {"id1": "protein_id"}, axis="columns"
    )  # Rename ID1 to protein_id

    # Dataframe for training data
    train_df = (
        df.loc[df["subset"] == "train"]
        .drop(columns=["subset"])
        .set_index("protein_id")
    )
    train_df = train_df.replace({"Y": {"pos": 1, "neg": 0}})

    # Dataframe for prediction data
    predict_df = (
        df.loc[df["subset"] == "test"]
        .drop(columns=["subset"])
        .set_index("protein_id")
    )
    predict_df = predict_df.replace({"Y": {"pos": -1, "neg": -1}})

    return train_df, predict_df


def save_train_predict_data(train, predict, tfile, pfile):
    """
    Save training and prediction data into pickle files.
    """
    print("Write training and prediction data to pickle files")
    # Write train data to a pickle file
    with open(tfile, "wb") as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Write prediction data to a pickle file
    with open(pfile, "wb") as handle:
        pickle.dump(predict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    """
    Using Oleg's RDS file, create pickle file containing features.
    """
    parser = argparse.ArgumentParser(
        description="Create features pickle file using RDS file"
    )
    parser.add_argument("--rdsfile", required=True, help="Input RDS file")
    parser.add_argument(
        "--trainfile",
        required=True,
        help="Output pickle file for training data",
    )
    parser.add_argument(
        "--predictfile",
        required=True,
        help="Output pickle file for predict data",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity"
    )

    args = parser.parse_args()

    # RDS to dataframe
    train_df, predict_df = extract_train_predict(args.rdsfile)

    # Write data to a pickle file
    save_train_predict_data(
        train_df, predict_df, args.trainfile, args.predictfile
    )
    print("Program executed successfully!!!")
