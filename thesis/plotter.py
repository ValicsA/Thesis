"""
Description

@author Ákos Valics
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

d = {
    "rl_0": {
        "time": np.array([]),
        "CO": np.array([]),
        "CO2": np.array([]),
        "NOx": np.array([]),
        "fuel": np.array([]),
        "HC": np.array([]),
        "noise": np.array([]),
        "PMx": np.array([]),
        "speed": np.array([]),
    },
    "rl_1": {
        "time": np.array([]),
        "CO": np.array([]),
        "CO2": np.array([]),
        "NOx": np.array([]),
        "fuel": np.array([]),
        "HC": np.array([]),
        "noise": np.array([]),
        "PMx": np.array([]),
        "speed": np.array([]),
    },
    "rl_2": {
        "time": np.array([]),
        "CO": np.array([]),
        "CO2": np.array([]),
        "NOx": np.array([]),
        "fuel": np.array([]),
        "HC": np.array([]),
        "noise": np.array([]),
        "PMx": np.array([]),
        "speed": np.array([]),
    },
}


with open("/home/akos/workspace/Thesis/thesis/data/highway_20200212-1458151581515895.21228-emission.csv") as results_csv:
    results = csv.reader(results_csv, delimiter=',')
    line_count = 0
    for row in results:
        if line_count == 0:
            columns_name = row
            line_count += 1
        else:
            if row[5] == "rl":
                d[row[6]]["time"] = np.append(d[row[6]]["time"], float(row[0]))
                d[row[6]]["CO"] = np.append(d[row[6]]["CO"], float(row[1]))
                d[row[6]]["CO2"] = np.append(d[row[6]]["CO2"], float(row[3]))
                d[row[6]]["NOx"] = np.append(d[row[6]]["NOx"], float(row[9]))
                d[row[6]]["fuel"] = np.append(d[row[6]]["fuel"], float(row[10]))
                d[row[6]]["HC"] = np.append(d[row[6]]["HC"], float(row[11]))
                d[row[6]]["noise"] = np.append(d[row[6]]["noise"], float(row[15]))
                d[row[6]]["PMx"] = np.append(d[row[6]]["PMx"], float(row[17]))
                d[row[6]]["speed"] = np.append(d[row[6]]["speed"], float(row[18]))

plt.plot(d["rl_0"]["time"], d["rl_0"]["fuel"])
# plt.yticks(np.arange(min(d["rl_0"]["fuel"]), max(d["rl_0"]["fuel"])+1, 0.5))
plt.show()
print("end: ", columns_name)
