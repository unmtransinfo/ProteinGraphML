import pandas as pd
import argparse


def read_python_file(pythonfile, mlimit):
    """
    Fetch the data from the output file of Python.
    """
    print("Sort the Python file in descending order by probability and select PID")
    df = pd.read_csv(pythonfile, sep='\t')
    df1 = df.sort_values(by='Predicted Probability', ascending=False)
    df2 = df1[['Protein Id', 'Symbol', 'Name', 'Predicted Probability']].iloc[:mlimit, :]  # Select top 100
    return df2.set_index('Protein Id').T.to_dict('list')


def read_r_file(rfile, mlimit):
    """
    Fetch the data from the output file of R.
    """
    print("Sort the Python file in descending order by probability and select PID")
    df = pd.read_csv(rfile, sep='\t')
    df1 = df.sort_values(by='pred.prob', ascending=False)
    df2 = df1[['protein_id', 'pred.prob']].iloc[:mlimit, :]		# Select top 100
    return df2.set_index('protein_id').T.to_dict('records')[0]


def data_for_correlation_coefficient(pdict, rdict):
    """
    Using the dictionaries created from R and Python files, create a dictionary for the
    computation of correlation coefficient.
    """
    print("Select data for correlation coefficient")
    dataValue = {'R_Prediction': [], 'Python_Prediction': []}
    recDict = {'Symbol': [], 'Name': [], 'R_prob': [], 'Python_prob': []}
    for k, v in rdict.items():
        if k in pdict:
            dataValue['R_Prediction'].append(v)
            dataValue['Python_Prediction'].append(pdict[k][2])
            recDict['Symbol'].append(pdict[k][0])
            recDict['Name'].append(pdict[k][1])
            recDict['R_prob'].append(v)
            recDict['Python_prob'].append(pdict[k][2])
    return recDict, dataValue


def find_correlation_coefficient(probs):
    """
    Find the Pearson correlation coefficient using the probabilities.
    """
    df = pd.DataFrame(data=probs)
    print("Pearson correlation coefficient:")
    print(df.corr(method='pearson'))


def save_data_in_tsv(rec, tsvfile):
    """
    Save the protein symbol, name and probabilities in a TSV file.
    """
    print("Save data in a TSV file")
    df = pd.DataFrame(rec)
    df.to_csv(tsvfile, sep="\t", index=False)


if __name__ == '__main__':
    """
    Start of the code
    """
    tsvFile = 'ATG_NEG_NO_LINCS/ATG_NEG_NO_LINCS_common_proteins.tsv'
    maxLimit = 100
    parser = argparse.ArgumentParser(description='Find correlation between R and Python predictions')
    parser.add_argument('--pythonfile', required=True, help='Python classification results file')
    parser.add_argument('--rfile', required=True, help='R classification results file')
    parser.add_argument('--tsvfile', default=tsvFile, help='File to save common proteins')
    parser.add_argument('--maxlimit', default=maxLimit, help='Max number of proteins to compare')
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

    # create dictionaries using R and Python files.
    args = parser.parse_args()
    myDictP = read_python_file(args.pythonfile, args.maxlimit)
    myDictR = read_r_file(args.rfile, args.maxlimit)
    recDict, dataValue = data_for_correlation_coefficient(myDictP, myDictR)

    # find Pearson correlation coefficient.
    find_correlation_coefficient(dataValue)

    # save the data in a tsv file.
    save_data_in_tsv(recDict, args.tsvfile)
    print("Program finished successfully")
