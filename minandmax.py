import csv
import pandas as pd

def getminmax(filename):
    min_values = []
    max_values = []
    counter = 0 
    filename = 'NEWDATASET.csv'
    with open(filename, 'r') as f:
        d_reader = csv.DictReader(f)
        headers = d_reader.fieldnames
        
    d_reader = pd.read_csv(filename)
    for i in range(0, len(headers)):
        min_values.append(d_reader[headers[i]].min())
        max_values.append(d_reader[headers[i]].max())

    min_values = min_values[1:-1]
    max_values = max_values[1:-1]
    
    return min_values, max_values
    



if __name__ == "__main__":
    print(hi)

    
