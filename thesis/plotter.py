"""
Description

@author Ákos Valics
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

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
        # plt.plot(self.d["rl_0"]["time"], self.d["rl_0"]["fuel"])
        # plt.plot(self.d["rl_1"]["time"], self.d["rl_1"]["fuel"])
        # plt.plot(self.d["rl_2"]["time"], self.d["rl_2"]["fuel"])
        # plt.plot(self.d["rl_0"]["time"], self.d["rl_0"]["fuel"])
        # plt.plot(self.d["rl_1"]["time"], self.d["rl_1"]["fuel"])
        # plt.plot(self.d["rl_2"]["time"], self.d["rl_2"]["fuel"])
        # plt.show()
        # Plot results in time
        for r_key, r_value in self.d_plot.items():
            plt.plot(self.d_plot["time"], r_value)
            plt.xlabel("time " + self.dimensions["time"])
            plt.ylabel(r_key + " " + self.dimensions[r_key])
            plt.title("time - " + r_key)
            plt.savefig(filename + "time_" + r_key)
            plt.show()

        # Plot x-y positions
        plt.plot(self.d["rl_0"]["x"], self.d["rl_0"]["y"])
        plt.plot(self.d["rl_1"]["x"], self.d["rl_1"]["y"])
        plt.plot(self.d["rl_2"]["x"], self.d["rl_2"]["y"])
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("x - y")
        plt.savefig(filename + "x_y")
        plt.show()

        plt.plot(self.d["rl_0"]["time"], self.d["rl_0"]["fuel"])
        plt.plot(self.d["rl_1"]["time"], self.d["rl_1"]["fuel"])
        plt.plot(self.d["rl_2"]["time"], self.d["rl_2"]["fuel"])
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("x - y")
        plt.savefig(filename + "fuel")
        plt.show()

        # for x_key, x_value in self.d_plot.items():
        #     ax = plt.axes()
        #     y_pos = np.arange(len(self.d_plot["speed"]))
        #     pl, x_value = zip(*sorted(zip(self.d_plot["speed"], x_value)))
        #     plt.bar(y_pos, x_value)
        #     # ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        #     plt.xticks(y_pos, pl)
        #     # ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        #     # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #     plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        #     ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        #     plt.title("speed - " + x_key)
        #     plt.show()

        for d_key, d_value in self.d_results.items():
            for d1_key, d1_value in d_value.items():
                for d2_key, d2_value in d1_value.items():
                    print(f"{d_key}'s {d1_key} {d2_key}: {self.d_results[d_key][d1_key][d2_key]}")
                    # print(f"{d_key}'s sum {d2_key}: {self.d_results[d_key]['sum'][d2_key]}")
                    with open("results.csv", mode="a") as write_results_file:
                        results_writer = csv.writer(write_results_file, delimiter=",", lineterminator="\n")
                        results_writer.writerow([d2_key, d2_value])
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
                # if counter % data_index == 0:
                means = np.append(means, [row[0], float(row[1])])
                # else:
                    # counter += 1
        means = np.reshape(means, (int(len(means)/2), 2))
        pl = np.array([])
        for i in range(means.shape[0]):
            if means[i, 0] == data:
                pl = np.append(pl, means[i, 1])
        pl = pl[1::2]
        pl = pl[:15]
        width = 0.2
        y_pos = np.arange(1, len(cases)+1)

        pl, y_pos = zip(*sorted(zip(pl, y_pos)))

        plt.bar(y_pos, pl)

        plt.xticks(y_pos)

        plt.show()

    def plot_difference(self, filename, data):
        barWidth = 0.2
        means = np.array([])
        with open(filename, mode="r") as results_csv:
            results = csv.reader(results_csv)
            for row in results:
                means = np.append(means, [row[0], float(row[1])])
        means = np.reshape(means, (int(len(means) / 2), 2))
        pl_data = np.array([])
        pl_speed = np.array([])
        for i in range(means.shape[0]):
            if means[i, 0] == data:
                pl_data = np.append(pl_data, means[i, 1])
            elif means[i, 0] == "speed":
                pl_speed = np.append(pl_speed, means[i, 1])
        pl_speed = pl_speed[::2]
        pl_speed = pl_speed[15:30]
        pl_data = pl_data[1::2]
        pl_data = pl_data[15:30]

        bar1 = pl_data[:5].astype(float)
        bar2 = pl_data[5:10].astype(float)
        bar3 = pl_data[10:15].astype(float)

        r1 = np.arange(len(bar1))
        r2 = [x2 + barWidth for x2 in r1]
        r3 = [x3 + barWidth*2 for x3 in r1]
        r4 = np.append(np.append(r1, r2), r3)

        pl_speed = np.round(pl_speed.astype(float), 1)
        pl_data1 = np.around(pl_data.astype(float)).astype(int)

        plt.bar(r1, bar1, width=barWidth, color='blue', edgecolor='black', label='20 m/s')
        plt.bar(r2, bar2, width=barWidth, color='cyan', edgecolor='black', label='30 m/s')
        plt.bar(r3, bar3, width=barWidth, color='red', edgecolor='black', label='40 m/s')

        for i in range(len(r4)):
            plt.text(x=r4[i]-0.07, y=pl_data1[i]+3700, s=f"v = {pl_speed[i]}", color='black', rotation=90)
        plt.xticks([r + barWidth for r in range(len(bar1))], ['700', '500', '300', '200', '100'])
        plt.ylabel("Value")
        plt.xlabel("Float rate [veh/h]")
        plt.title(f"{data}")
        plt.legend()

        plt.show()

        pl_data = pl_data.astype(float)
        pl_speed = pl_speed.astype(float)
        pl_speed, pl_data = zip(*sorted(zip(pl_speed, pl_data)))

        plt.plot(pl_speed, pl_data)
        plt.ylabel(f"{data} {self.dimensions[data]}")
        plt.xlabel("Speed [m/s]")
        plt.title(f"{data}")

        plt.show()


def main(mode):
    plotter = Plotter()
    plotter.read_file(
        filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/highway_case_4_emission.csv")
    plotter.calculate_average()
    plotter.plot_results(
        filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/plots/Plot_Case_000_")
    # if mode == "generate":
    #     for a in range(1, 46):
    #         plotter = Plotter()
    #         plotter.read_file(filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/highway_case_{a}_emission.csv")
    #         plotter.calculate_average()
    #         plotter.plot_results(filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/plots/Plot_Case_{a}_")
    #
    # elif mode == "plot":
    #     filename = "/home/akos/workspace/Thesis/thesis/results_all.csv"
    #     cases = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15"]
    #     plotter = Plotter()
    #     # plotter.plot_cases(filename=filename, cases=cases, data="fuel")
    #     plotter.plot_difference(filename=filename, data="fuel")


if __name__ == "__main__":

    main(mode="plot")

    print("End")
