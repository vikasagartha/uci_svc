import csv
from numpy import array 

def read(fname):
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        col_names = []
        data = []
        for row in csv_reader:
            if line_count == 0:
                col_names = row
                line_count += 1
            else:
                data.append([int(float(x)) for x in row])
                line_count += 1
        return {'col_names': col_names,
                'features': [x[:-1] for x in data],
                'labels': [x[-1] for x in data]}
