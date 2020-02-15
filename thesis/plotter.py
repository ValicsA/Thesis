"""
Description

@author Ákos Valics
"""
import csv
import numpy as np

with open("/home/akos/workspace/Thesis/thesis/data/highway_20200212-1458151581515895.21228-emission.csv") as results_csv:
    results = csv.reader(results_csv, delimiter=',')
    line_count = 0
    for row in results:
        if line_count == 0:
            columns_name = row
            res = np.array([])
            line_count += 1
        else:
            columns_value = np.asarray(row)
            res = np.concatenate(res, columns_name)

print("end: ", results)
