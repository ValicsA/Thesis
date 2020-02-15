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
d_results = {
    "avr": {
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
    "sum": {
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
    "avr_all": {
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
    "sum_all": {
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

# for avr_step in range(min(d["rl_0"]["time"].size, d["rl_1"]["time"].size, d["rl_2"]["time"].size)):
#     d["rl_avr"]["time"][avr_step] =
start = int(max(min(d["rl_0"]["time"]), min(d["rl_1"]["time"]), min(d["rl_2"]["time"])))
end = int(min(max(d["rl_0"]["time"]), max(d["rl_1"]["time"]), max(d["rl_2"]["time"])))

results_average = {
    "time": (d["rl_0"]["time"][start:end] + d["rl_1"]["time"][start:end] + d["rl_2"]["time"][start:end]) / 3,
    "CO": (d["rl_0"]["CO"][start:end] + d["rl_1"]["CO"][start:end] + d["rl_2"]["CO"][start:end]) / 3,
    "CO2": (d["rl_0"]["CO2"][start:end] + d["rl_1"]["CO2"][start:end] + d["rl_2"]["CO2"][start:end]) / 3,
    "NOx": (d["rl_0"]["NOx"][start:end] + d["rl_1"]["NOx"][start:end] + d["rl_2"]["NOx"][start:end]) / 3,
    "fuel": (d["rl_0"]["fuel"][start:end] + d["rl_1"]["fuel"][start:end] + d["rl_2"]["fuel"][start:end]) / 3,
    "HC": (d["rl_0"]["HC"][start:end] + d["rl_1"]["HC"][start:end] + d["rl_2"]["HC"][start:end]) / 3,
    "noise": (d["rl_0"]["noise"][start:end] + d["rl_1"]["noise"][start:end] + d["rl_2"]["noise"][start:end]) / 3,
    "PMx": (d["rl_0"]["PMx"][start:end] + d["rl_1"]["PMx"][start:end] + d["rl_2"]["PMx"][start:end]) / 3,
    "speed": (d["rl_0"]["speed"][start:end] + d["rl_1"]["speed"][start:end] + d["rl_2"]["speed"][start:end]) / 3,
}
for d_key, d_value in d.items():
    for key, value in d_value.items():
        d_results["avr"][key] = np.average(value)
        d_results["sum"][key] = np.sum(value)
        print(f"{d_key}'s average {key}: {d_results['avr'][key]}")
        print(f"{d_key}'s sum {key}: {d_results['sum'][key]}")

# plt.plot(d["rl_0"]["time"], d["rl_0"]["fuel"])
# plt.plot(d["rl_1"]["time"], d["rl_1"]["fuel"])
# plt.plot(d["rl_2"]["time"], d["rl_2"]["fuel"])
plt.plot(results_average["time"], results_average["fuel"])
# plt.yticks(np.arange(min(d["rl_0"]["fuel"]), max(d["rl_0"]["fuel"])+1, 0.5))
plt.show()

for r_key, r_value in results_average.items():
    d_results["avr_all"][r_key] = np.average(r_value)
    d_results["sum_all"][r_key] = np.sum(r_value)
    print(f"Overall average {r_key} is {d_results['avr_all'][r_key]}")
    print(f"Overall average sum {r_key} is {d_results['sum_all'][r_key]}")

print("end: ", columns_name)
