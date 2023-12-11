import pandas as pd


def confronta_csv(file1, file2):
    # Carica i file CSV in due DataFrame
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Trova le differenze tra i due DataFrame
    diff_df = df1.compare(df2)

    # Stampa le differenze
    if diff_df.empty:
        print("I due CSV sono identici.")
    else:
        print("Differenze tra i due CSV:")
        for index, row in diff_df.iterrows():
            print(f"Riga {index}:\n{row}\n")


confronta_csv('../output_files/XGBoostSubmissionReranked.csv', '../output_files/XGBoostSubmission.csv')
