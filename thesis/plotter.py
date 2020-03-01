"""
Description

@author Ákos Valics
"""
import csv
import numpy as np
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self):
        self.basic_dict = {
                "time": np.array([]),
                "CO": np.array([]),
                "CO2": np.array([]),
                "NOx": np.array([]),
                "fuel": np.array([]),
                "HC": np.array([]),
                "noise": np.array([]),
                "PMx": np.array([]),
                "speed": np.array([]),
                "y": np.array([]),
                "x": np.array([]),
            }

        self.d = {
            "rl_0": self.basic_dict.copy(),
            "rl_1": self.basic_dict.copy(),
            "rl_2": self.basic_dict.copy(),
        }
        self.d_results = {
            "rl_0": {
                "avr": self.basic_dict.copy(),
                "sum": self.basic_dict.copy(),
            },
            "rl_1": {
                "avr": self.basic_dict.copy(),
                "sum": self.basic_dict.copy(),
            },
            "rl_2": {
                "avr": self.basic_dict.copy(),
                "sum": self.basic_dict.copy(),
            },
        }
        self.d_results_all = {
            "avr_all": self.basic_dict.copy(),
            "sum_all": self.basic_dict.copy(),
        }
        self.dimensions = {
            "time": "[s]",
            "CO": "[mg/s]",
            "CO2": "[mg/s]",
            "NOx": "[mg/s]",
            "fuel": "[ml/s]",
            "HC": "[mg/s]",
            "noise": "[dB]",
            "PMx": "[mg/s]",
            "speed": "[m/s]",
            "y": "[m]",
            "x": "[m]",
        }
        self.d_plot = self.basic_dict.copy()

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
                        self.d[row[6]]["y"] = np.append(self.d[row[6]]["y"], float(row[2]))
                        self.d[row[6]]["x"] = np.append(self.d[row[6]]["x"], float(row[12]))

    def calculate_average(self):
        start = int(max(min(self.d["rl_0"]["time"]), min(self.d["rl_1"]["time"]), min(self.d["rl_2"]["time"])))
        end = int(min(max(self.d["rl_0"]["time"]), max(self.d["rl_1"]["time"]), max(self.d["rl_2"]["time"])))
        diff = end - start

        for f_key, f_value in self.d.items():
            for f1_key, f1_value in f_value.items():
                self.d[f_key][f1_key] = f1_value[:diff]

        for p_key, p_value in self.d_plot.items():
            self.d_plot[p_key] = (self.d["rl_0"][p_key] + self.d["rl_1"][p_key] + self.d["rl_2"][p_key]) / 3

        for d_key, d_value in self.d.items():
            for key, value in d_value.items():
                self.d_results[d_key]["avr"][key] = np.average(value)
                self.d_results[d_key]["sum"][key] = np.sum(value)

        for key, value in self.basic_dict.items():
            self.d_results_all["avr_all"][key] = np.average([self.d_results["rl_0"]["avr"][key], self.d_results["rl_1"]["avr"][key], self.d_results["rl_2"]["avr"][key]])
            self.d_results_all["sum_all"][key] = np.sum([self.d_results["rl_0"]["sum"][key], self.d_results["rl_1"]["sum"][key], self.d_results["rl_2"]["sum"][key]])

        # for r_key, r_value in self.results_average.items():
        #     self.d_results["avr_all"][r_key] = np.average(r_value)
        #     self.d_results["sum_all"][r_key] = np.sum(r_value)

    def plot_results(self, filename):
        # plt.plot(self.d["rl_0"]["time"], self.d["rl_0"]["y"])
        # plt.plot(self.d["rl_1"]["time"], self.d["rl_1"]["y"])
        # plt.plot(self.d["rl_2"]["time"], self.d["rl_2"]["y"])
        # plt.plot(self.d["rl_0"]["time"], self.d["rl_0"]["x"])
        # plt.plot(self.d["rl_1"]["time"], self.d["rl_1"]["x"])
        # plt.plot(self.d["rl_2"]["time"], self.d["rl_2"]["x"])
        for r_key, r_value in self.d_plot.items():
            plt.plot(self.d_plot["time"], r_value)
            plt.xlabel("time " + self.dimensions["time"])
            plt.ylabel(r_key + " " + self.dimensions[r_key])
            plt.title("time - " + r_key)
            plt.savefig(filename + "time_" + r_key)
            # plt.yticks(np.arange(min(d["rl_0"]["fuel"]), max(d["rl_0"]["fuel"])+1, 0.5))
            plt.show()

        for d_key, d_value in self.d_results.items():
            for d1_key, d1_value in d_value.items():
                for d2_key, d2_value in d1_value.items():
                    print(f"{d_key}'s {d1_key} {d2_key}: {self.d_results[d_key][d1_key][d2_key]}")
                    # print(f"{d_key}'s sum {d2_key}: {self.d_results[d_key]['sum'][d2_key]}")
                    with open("results.csv", mode="a") as write_results_file:
                        results_writer = csv.writer(write_results_file, delimiter=",", lineterminator="\n")
                        results_writer.writerow([self.d_results[d_key][d1_key][d2_key]])
                        # results_writer.writerow([self.d_results['sum'][key]])
                print("*************************************************************************************")
        print("*************************************************************************************")

        for r_key, r_value in self.d_results_all.items():
            for r1_key, r1_value in r_value.items():
                print(f"{r_key} {r1_key} is {self.d_results_all[r_key][r1_key]}")
                # print(f"Overall average sum {r_key} is {self.d_results['sum_all'][r_key]}")
                with open("results_all.csv", mode="a") as write_results_file:
                    results_writer = csv.writer(write_results_file, delimiter=",", lineterminator="\n")
                    results_writer.writerow([r1_key, r1_value])
                    # results_writer.writerow([self.d_results['sum_all'][r_key]])

    def plot_cases(self, filename, cases, data):

        means = np.array([])
        with open(filename, mode="r") as results_csv:
            results = csv.reader(results_csv)

            for row in results:
                if counter % data_index == 0:
                    means = np.append(means, row)
                else:
                    counter += 1
        means = means[:len(cases)]
        width = 0.2
        x = np.arange(len(cases))

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, means, width, label='fuel')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # autolabel(rects1)

        fig.tight_layout()

        plt.show()


def main(mode):
    if mode == "generate":
        for a in range(1, 46):
            plotter = Plotter()
            plotter.read_file(filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/highway_case_{a}_emission.csv")
            plotter.calculate_average()
            plotter.plot_results(filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/plots/Plot_Case_{a}_")

    elif mode == "plot":
        filename = "/home/akos/workspace/Thesis/thesis/results.csv"
        cases = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15"]
        plotter = Plotter()
        plotter.plot_cases(filename=filename, cases=cases, data_index=4)


if __name__ == "__main__":

    main(mode="plot")

    print("End")
