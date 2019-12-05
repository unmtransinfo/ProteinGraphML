import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


def read_python_file(pythonfile):
    """
    Fetch the data from the output file of Python
    """
    print("Sort the Python file in descending order by probability and select PID")
    df = pd.read_csv(pythonfile, sep='\t')
    df1 = df.sort_values(by='Predicted Probability', ascending=False)
    return list(df1['Protein Id'])


def read_r_file(rfile):
    """
    Fetch the data from the output file of R.
    """
    print("Sort the Python file in descending order by probability and select PID")
    df = pd.read_csv(rfile, sep='\t')
    df1 = df.sort_values(by='pred.prob', ascending=False)
    return list(df1['protein_id'])


def find_common_pid(searchList, RList, pythonList):
    """
    Find the common pid in R and Python list
    """
    print('How many PIDs are common in R and Python outcomes')
    commonInBoth = []
    for v in searchList:
        vals = np.intersect1d(RList[:v], pythonList[:v])
        commonInBoth.append(round(len(vals) * 100 / v, 2))
    return commonInBoth


def draw_plot(x, train, predict, imgFile):
    """
    draw plot using the given data
    """
    print("Draw plot using the data")
    fig, ax = plt.subplots()
    ax.plot(x, predict, label='predict data')
    if train is not None:
        ax.plot(x, train, label='train data')
    ax.set(xlabel='Top N', ylabel='Common pid',
           title='Number of common pid in R and Python prediction')
    ax.grid(b=True, which='major', color='c', linestyle='-')
    ax.grid(b=True, which='minor', color='r', linestyle='--')
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
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


def draw_bar_plot(xdataTr, xdataPr, train, predict, imgfile):
    """
    Draw bar chart using the given data.
    """
    xTr = np.arange(len(xdataTr))  # the label locations
    xPr = np.arange(len(xdataPr))  # the label locations
    width = 0.40  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(xTr - width / 2, train, width, label='Training data')
    rects2 = ax.bar(xPr + width / 2, predict, width, label='Predict data')

    ax.set_ylabel('%Common PID')
    ax.set_xlabel('Top N')
    ax.set_title('Common pid in R and Python predictions')
    ax.set_xticks(xPr)
    ax.set_xticklabels(xdataPr)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    '''
    start of the code
    '''
    imgFile = 'ATG_NEG_NO_LINCS/ATG_NEG_NO_LINCS_common1.png'
    maxLimit = 1100
    parser = argparse.ArgumentParser(description='Find what percentage of PID are common in R and Python predictions')
    parser.add_argument('--pythonTr', required=True, help='Python classification results file for training data')
    parser.add_argument('--pythonPr', required=True, help='Python classification results file for prediction data')
    parser.add_argument('--rTr', required=True, help='R classification results file for training data')
    parser.add_argument('--rPr', required=True, help='R classification results file for prediction data')
    parser.add_argument('--imgfile', default=imgFile, help='Filename to save the plot')
    parser.add_argument('--maxlimit', default=maxLimit, help='Max number of proteins to compare')
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

    args = parser.parse_args()

    # find common PID in training data
    pythonListTr = read_python_file(args.pythonTr)
    RListTr = read_r_file(args.rTr)
    n = round(len(pythonListTr), -2) + 200 if len(pythonListTr) > round(len(pythonListTr), -2) else round(
        len(pythonListTr), -2) + 100
    searchListTr = [i for i in range(100, n, 100)]
    commonInTr = find_common_pid(searchListTr, RListTr, pythonListTr)
    commonInTr[-1] = 100.0  # for training data, the last element should be 100%

    # find common PID in prediction data
    pythonListPr = read_python_file(args.pythonPr)
    RListPr = read_r_file(args.rPr)
    searchListPr = [i for i in range(100, args.maxlimit, 100)]
    commonInPr = find_common_pid(searchListPr, RListPr, pythonListPr)

    # Draw plot using the data
    # draw_plot(searchList, commonInTr, commonInPr, imgFile)
    draw_bar_plot(searchListTr, searchListPr, commonInTr, commonInPr, imgFile)
