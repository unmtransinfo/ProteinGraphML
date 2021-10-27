import argparse
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_python_file(pythonfile):
    """
    Fetch the data from the output file of Python
    """
    print(
        "Sort the Python file in descending order by probability and select PID"
    )
    df = pd.read_csv(pythonfile, sep="\t")
    df1 = df.sort_values(by="Predicted Probability", ascending=False)
    return list(df1["Protein Id"])


def find_common_pid(searchList, oleglist, tcrdList):
    """
    Find the common pid in R and Python list
    """
    print("Fnd common PIDs in R and Python outcomes")
    commonInBoth = []
    for v in searchList:
        vals = np.intersect1d(oleglist[:v], tcrdList[:v])
        commonInBoth.append(round(len(vals) * 100 / v, 2))
    return commonInBoth


def draw_plot(x, train, predict, imgFile):
    """
    draw plot using the given data
    """
    print("Draw plot using the data")
    fig, ax = plt.subplots()
    ax.plot(x, predict, label="predict data")
    if train is not None:
        ax.plot(x, train, label="train data")
    ax.set(
        xlabel="Top N",
        ylabel="Common pid",
        title="Number of common pid in R and Python prediction",
    )
    ax.grid(b=True, which="major", color="c", linestyle="-")
    ax.grid(b=True, which="minor", color="r", linestyle="--")
    ax.legend()
    ax.set_xticks(x)
    plt.minorticks_on()
    fig.savefig(imgFile)
    plt.show()


def autolabel(rects, ax):
    """
    Attach a text label above each bar in rects, displaying its values.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
        )


def draw_bar_plot(xdataTr, xdataPr, train, predict, imgfile):
    """
    Draw bar chart using the given data.
    """
    print("Draw bar plot using the data")
    xTr = np.arange(len(xdataTr))  # the label locations
    xPr = np.arange(len(xdataPr))  # the label locations
    width = 0.40  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(xTr - width / 2, train, width, label="Training data")
    rects2 = ax.bar(xPr + width / 2, predict, width, label="Predict data")

    ax.set_ylabel("%Common PID")
    ax.set_xlabel("Top N")
    ax.set_title(
        "\n".join(
            wrap(
                "Common protein Ids in classifications using TCRD and OlegDB (Autophagy)",
                60,
            )
        )
    )
    ax.set_xticks(xPr)
    ax.set_xticklabels(xdataPr)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    fig.savefig(imgfile)
    plt.show()


if __name__ == "__main__":
    """
    start of the code
    """
    maxLimit = 1100
    parser = argparse.ArgumentParser(
        description="Find what percentage of PID are common in R and Python predictions"
    )
    parser.add_argument(
        "--tcrdTr",
        required=True,
        help="Classification results file for training data using "
        "TCRD database",
    )
    parser.add_argument(
        "--tcrdPr",
        required=True,
        help="Classification results file for prediction data using "
        "TCRD database",
    )
    parser.add_argument(
        "--olegTr",
        required=True,
        help="Classification results file for training data using "
        "OlegDB database",
    )
    parser.add_argument(
        "--olegPr",
        required=True,
        help="R classification results file for prediction data "
        "using OlegDB database",
    )
    parser.add_argument("--imgfile", help="Filename to save the plot")
    parser.add_argument(
        "--maxlimit",
        default=maxLimit,
        type=int,
        help="Max number of proteins to compare",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity"
    )

    args = parser.parse_args()

    # find common PID in training data
    TcrdListTr = read_python_file(args.tcrdTr)
    OlegListTr = read_python_file(args.olegTr)
    n = (
        round(len(TcrdListTr), -2) + 200
        if len(TcrdListTr) > round(len(TcrdListTr), -2)
        else round(len(TcrdListTr), -2) + 100
    )
    if n > 1000:
        n = 1000
    searchListTr = [i for i in range(100, n, 100)]
    commonInTr = find_common_pid(searchListTr, OlegListTr, TcrdListTr)
    commonInTr[
        -1
    ] = 100.0  # for training data, the last element should be 100%

    # find common PID in prediction data
    TcrdListPr = read_python_file(args.tcrdPr)
    OlegListPr = read_python_file(args.olegPr)
    searchListPr = [i for i in range(100, args.maxlimit + 100, 100)]
    commonInPr = find_common_pid(searchListPr, OlegListPr, TcrdListPr)

    # Draw plot using the data
    # draw_plot(searchList, commonInTr, commonInPr, imgFile)
    draw_bar_plot(
        searchListTr, searchListPr, commonInTr, commonInPr, args.imgfile
    )
    print("Program finished successfully!!!")
