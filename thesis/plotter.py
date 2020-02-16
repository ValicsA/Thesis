"""
Description

@author Ákos Valics
"""
import csv
import numpy as np
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self):
        basic_dict = {
                "time": np.array([]),
                "CO": np.array([]),
                "CO2": np.array([]),
                "NOx": np.array([]),
                "fuel": np.array([]),
                "HC": np.array([]),
                "noise": np.array([]),
                "PMx": np.array([]),
                "speed": np.array([]),
            }

        self.d = {
            "rl_0": basic_dict.copy(),
            "rl_1": basic_dict.copy(),
            "rl_2": basic_dict.copy(),
        }
        self.d_results = {
            "avr": basic_dict.copy(),
            "sum": basic_dict.copy(),
            "avr_all": basic_dict.copy(),
            "sum_all": basic_dict.copy(),
        }
        self.results_average = {}

    def read_file(self, filename):
        with open(filename) as results_csv:
            results = csv.reader(results_csv, delimiter=',')
            line_count = 0
            for row in results:
                if line_count == 0:
                    columns_name = row
                    print(f"The measured values are: {columns_name}")
                    line_count += 1
                else:
                    if row[5] == "rl":
                        self.d[row[6]]["time"] = np.append(self.d[row[6]]["time"], float(row[0]))
                        self.d[row[6]]["CO"] = np.append(self.d[row[6]]["CO"], float(row[1]))
                        self.d[row[6]]["CO2"] = np.append(self.d[row[6]]["CO2"], float(row[3]))
                        self.d[row[6]]["NOx"] = np.append(self.d[row[6]]["NOx"], float(row[9]))
                        self.d[row[6]]["fuel"] = np.append(self.d[row[6]]["fuel"], float(row[10]))
                        self.d[row[6]]["HC"] = np.append(self.d[row[6]]["HC"], float(row[11]))
                        self.d[row[6]]["noise"] = np.append(self.d[row[6]]["noise"], float(row[15]))
                        self.d[row[6]]["PMx"] = np.append(self.d[row[6]]["PMx"], float(row[17]))
                        self.d[row[6]]["speed"] = np.append(self.d[row[6]]["speed"], float(row[18]))

    def calculate_average(self):
        start = int(max(min(self.d["rl_0"]["time"]), min(self.d["rl_1"]["time"]), min(self.d["rl_2"]["time"])))
        end = int(min(max(self.d["rl_0"]["time"]), max(self.d["rl_1"]["time"]), max(self.d["rl_2"]["time"])))

        self.results_average = {
            "time": (self.d["rl_0"]["time"][start:end] + self.d["rl_1"]["time"][start:end] + self.d["rl_2"]["time"][start:end]) / 3,
            "CO": (self.d["rl_0"]["CO"][start:end] + self.d["rl_1"]["CO"][start:end] + self.d["rl_2"]["CO"][start:end]) / 3,
            "CO2": (self.d["rl_0"]["CO2"][start:end] + self.d["rl_1"]["CO2"][start:end] + self.d["rl_2"]["CO2"][start:end]) / 3,
            "NOx": (self.d["rl_0"]["NOx"][start:end] + self.d["rl_1"]["NOx"][start:end] + self.d["rl_2"]["NOx"][start:end]) / 3,
            "fuel": (self.d["rl_0"]["fuel"][start:end] + self.d["rl_1"]["fuel"][start:end] + self.d["rl_2"]["fuel"][start:end]) / 3,
            "HC": (self.d["rl_0"]["HC"][start:end] + self.d["rl_1"]["HC"][start:end] + self.d["rl_2"]["HC"][start:end]) / 3,
            "noise": (self.d["rl_0"]["noise"][start:end] + self.d["rl_1"]["noise"][start:end] + self.d["rl_2"]["noise"][start:end]) / 3,
            "PMx": (self.d["rl_0"]["PMx"][start:end] + self.d["rl_1"]["PMx"][start:end] + self.d["rl_2"]["PMx"][start:end]) / 3,
            "speed": (self.d["rl_0"]["speed"][start:end] + self.d["rl_1"]["speed"][start:end] + self.d["rl_2"]["speed"][start:end]) / 3,
        }

        for d_key, d_value in self.d.items():
            for key, value in d_value.items():
                self.d_results["avr"][key] = np.average(value)
                self.d_results["sum"][key] = np.sum(value)

        for r_key, r_value in self.results_average.items():
            self.d_results["avr_all"][r_key] = np.average(r_value)
            self.d_results["sum_all"][r_key] = np.sum(r_value)

    def plot_results(self):
        # plt.plot(d["rl_0"]["time"], d["rl_0"]["fuel"])
        # plt.plot(d["rl_1"]["time"], d["rl_1"]["fuel"])
        # plt.plot(d["rl_2"]["time"], d["rl_2"]["fuel"])
        plt.plot(self.results_average["time"], self.results_average["fuel"])
        # plt.yticks(np.arange(min(d["rl_0"]["fuel"]), max(d["rl_0"]["fuel"])+1, 0.5))
        plt.show()

        for d_key, d_value in self.d.items():
            for key, value in d_value.items():
                print(f"{d_key}'s average {key}: {self.d_results['avr'][key]}")
                print(f"{d_key}'s sum {key}: {self.d_results['sum'][key]}")
            print("*************************************************************************************")
        print("*************************************************************************************")

        for r_key, r_value in self.results_average.items():
            print(f"Overall average {r_key} is {self.d_results['avr_all'][r_key]}")
            print(f"Overall average sum {r_key} is {self.d_results['sum_all'][r_key]}")


def main():
    plotter = Plotter()
    plotter.read_file(filename="/home/akos/workspace/Thesis/thesis/data/highway_20200212-1458151581515895.21228-emission.csv")
    plotter.calculate_average()
    plotter.plot_results()


if __name__ == "__main__":
    main()
    print("End")
