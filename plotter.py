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

    def __init__(self, folder_name):
        """
        Initializer of Plotter class. Creates class variables.

        :param folder_name: Name of the folder where the measurements are located.
        """
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
        self.folder_name = folder_name

    def read_file(self, filename):
        """
        Reads the measurement file and saves the relevant results to a dictionary.

        :param filename: Path and name of the measurement file.

        :return None
        """
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

    def fix_zero_emissions(self):
        """
        Corrects zero emission values in measurement. Multiplies the previous value by 0.99.
        """
        for vehicles_keys, vehicles_values in self.d.items():
            for keys, values in vehicles_values.items():
                for i in range(1, len(values)):
                    self.d[vehicles_keys][keys][i] = self.d[vehicles_keys][keys][i]\
                        if self.d[vehicles_keys][keys][i] > 0 else self.d[vehicles_keys][keys][i-1]*0.99

    def calculate_average(self):
        """
        Calculates the average values for every rl vehicle respectively and for all vehicles too.
        self.d_results contains average values for every vehicle respectively.
        self.d_results_all contains average values for all vehicles.
        """
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
            self.d_results_all["avr_all"][key] = np.average([self.d_results["rl_0"]["avr"][key],
                                                             self.d_results["rl_1"]["avr"][key],
                                                             self.d_results["rl_2"]["avr"][key]])
            self.d_results_all["sum_all"][key] = np.average([self.d_results["rl_0"]["sum"][key],
                                                             self.d_results["rl_1"]["sum"][key],
                                                             self.d_results["rl_2"]["sum"][key]])

        # for r_key, r_value in self.results_average.items():
        #     self.d_results["avr_all"][r_key] = np.average(r_value)
        #     self.d_results["sum_all"][r_key] = np.sum(r_value)

    def plot_results(self, filename):
        """
        Plots the results and save the results to a folder.
        """
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
        plt.xlabel("time [s]")
        plt.ylabel("fuel [ml/s]")
        plt.title("fuel - time")
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
                    with open(f"/home/akos/workspace/Thesis/thesis/data/{self.folder_name}/results.csv", mode="a")\
                            as write_results_file:
                        results_writer = csv.writer(write_results_file, delimiter=",", lineterminator="\n")
                        results_writer.writerow([d2_key, d2_value])
                        # results_writer.writerow([self.d_results['sum'][key]])
                print("*************************************************************************************")
        print("*************************************************************************************")

        for r_key, r_value in self.d_results_all.items():
            for r1_key, r1_value in r_value.items():
                print(f"{r_key} {r1_key} is {self.d_results_all[r_key][r1_key]}")
                # print(f"Overall average sum {r_key} is {self.d_results['sum_all'][r_key]}")
                with open(f"/home/akos/workspace/Thesis/thesis/data/{self.folder_name}/results_all.csv", mode="a")\
                        as write_results_file:
                    results_writer = csv.writer(write_results_file, delimiter=",", lineterminator="\n")
                    results_writer.writerow([r1_key, r1_value])
                    # results_writer.writerow([self.d_results['sum_all'][r_key]])

    def plot_cases(self, filename, cases, data):
        """
        Plots the data variable in different test cases.
        """
        means = self._read_mean_results(filename=filename)
        pl = np.array([])
        for i in range(means.shape[0]):
            if means[i, 0] == data:
                pl = np.append(pl, means[i, 1])
        pl = pl[1::2]
        pl = pl[:4]
        width = 0.2
        y_pos = np.arange(1, len(cases)+1)

        pl, y_pos = zip(*sorted(zip(pl, y_pos)))

        plt.bar(y_pos, pl)

        plt.xticks(y_pos)

        plt.show()

    @ staticmethod
    def _read_mean_results(filename):
        """
        Reads results from a file and loads them into a numpy array.
        """
        means = np.array([])
        with open(filename, mode="r") as results_csv:
            results = csv.reader(results_csv)
            for row in results:
                means = np.append(means, [row[0], float(row[1])])
        means = np.reshape(means, (int(len(means) / 2), 2))
        return means

    @ staticmethod
    def _return_plotdata_plotspeed(means, start_index, data_type, data_name_dim_exp):
        pl_data = np.array([])
        pl_speed = np.array([])
        for i in range(means.shape[0]):
            if means[i, 0] == data_name_dim_exp[0]:
                pl_data = np.append(pl_data, means[i, 1])
            elif means[i, 0] == "speed":
                pl_speed = np.append(pl_speed, means[i, 1])
        pl_speed = pl_speed[::2]
        pl_speed = pl_speed[start_index:start_index + 12]
        pl_data = pl_data[::2] if data_type == "Avr" else pl_data[1::2]
        pl_data = pl_data[start_index:start_index + 12]
        return pl_data.astype(float), pl_speed.astype(float)

    def plot_difference(self, filename, start_index, data_type, data_name_dim_exp, save_path):
        """
        Plots the data_type (Sum or Avr) data_name_dim_exp emissions.
        Creates four bars for different number of traffic vehicles, and three groups of these four bars for
            desired RL speeds. Draws the actual speed on the top of each bars.
        Saves the generated plots to save_path folder with a specific name.
        """
        bar_width = 0.2
        means = self._read_mean_results(filename=filename)
        pl_data, pl_speed = self._return_plotdata_plotspeed(means=means, start_index=start_index, data_type=data_type,
                                                            data_name_dim_exp=data_name_dim_exp)

        bar1 = pl_data[:3]
        bar2 = pl_data[3:6]
        bar3 = pl_data[6:9]
        bar4 = pl_data[9:12]

        r1 = np.arange(len(bar1))
        r2 = [x2 + bar_width for x2 in r1]
        r3 = [x3 + bar_width*2 for x3 in r1]
        r4 = [x4 + bar_width*3 for x4 in r1]
        r5 = np.append(np.append(np.append(r1, r2), r3), r4)

        pl_speed = np.round(pl_speed, 1)

        plt.bar(r1, bar1, width=bar_width, color='yellow', edgecolor='black', label='10 t.v.')
        plt.bar(r2, bar2, width=bar_width, color='cyan', edgecolor='black', label='15 t.v.')
        plt.bar(r3, bar3, width=bar_width, color='red', edgecolor='black', label='20 t.v.')
        plt.bar(r4, bar4, width=bar_width, color='pink', edgecolor='black', label='25 t.v.')

        for i in range(len(r5)):
            plt.text(x=r5[i]-bar_width/2.3, y=pl_data[i], s=f"v={pl_speed[i]}", color='black', rotation=0, fontsize=6)
        plt.xticks([r + bar_width*1.5 for r in range(len(bar1))], ['20', '30', '35'])
        plt.ylabel(f"{data_name_dim_exp[0]} {data_name_dim_exp[1]}")
        plt.xlabel(f"Desired speed of RL vehicles [m/s]")
        if start_index == 0:
            tv_speed = 15
        elif start_index == 12:
            tv_speed = 20
        else:
            tv_speed = 30
        plt.title(f"{data_type} {data_name_dim_exp[0]} {data_name_dim_exp[2]} - {tv_speed} m/s t.v. speed")
        plt.legend()
        plt.savefig(f"{save_path}{data_name_dim_exp[0]}_tvs{tv_speed}_{data_type}")

        plt.show()

    @ staticmethod
    def _fit_line(x, y, exp, legend):
        x_, y_ = zip(*sorted(zip(x, y)))
        coeff = np.polyfit(x_, y_, exp)
        poly = np.poly1d(coeff)
        new_x = np.linspace(x_[0], x_[-1])
        new_y = poly(new_x)
        plt.plot(x_, y_, 'o', new_x, new_y, label=legend)

    def plot_values_and_speed(self, filename, start_index, data_type, data_name_dim_exp, save_path):
        means = self._read_mean_results(filename=filename)
        pl_data, pl_speed = self._return_plotdata_plotspeed(means=means, start_index=start_index, data_type=data_type,
                                                            data_name_dim_exp=data_name_dim_exp)

        self._fit_line(x=pl_speed[:4], y=pl_data[:4], exp=2, legend="20 m/s")
        self._fit_line(x=pl_speed[4:8], y=pl_data[4:8], exp=2, legend="30 m/s")
        self._fit_line(x=pl_speed[8:12], y=pl_data[8:12], exp=2, legend="35 m/s")

        if start_index == 0:
            tv_speed = 15
        elif start_index == 12:
            tv_speed = 20
        else:
            tv_speed = 30

        plt.legend()
        plt.ylabel(f"{data_name_dim_exp[0]} {data_name_dim_exp[1]}")
        plt.xlabel("Speed [m/s]")
        plt.title(f"{data_type} {data_name_dim_exp[0]} {data_name_dim_exp[2]} - {tv_speed} t.v. speed")
        plt.savefig(f"{save_path}{data_name_dim_exp[0]}_speed_{data_type}_{tv_speed}")
        plt.show()

    @staticmethod
    def _cut_relevant_results(data, data_type, cut_from_bas, cut_from_add):
        start = 0 if data_type == "avr" else 1
        data = data[start::2]
        data_add = data[45:]
        data_add = data_add[cut_from_add::3]
        data_bas = data[:45]
        data_bas = data_bas[cut_from_bas::5]
        data_plot = np.concatenate((data_bas, data_add))
        return data_plot

    def plot_same_flow_different_cases(self, filename, data):
        x_label_bas = range(1, 46)
        x_label_bas = x_label_bas[2::5]
        x_label_add = range(46, 82)
        x_label_add = x_label_add[1::3]
        x_label = np.concatenate((x_label_bas, x_label_add))
        bar_width = 0.3
        means = self._read_mean_results(filename=filename)
        pl_data = np.array([])
        pl_speed = np.array([])
        for i in range(means.shape[0]):
            if means[i, 0] == data:
                pl_data = np.append(pl_data, float(means[i, 1]))
            elif means[i, 0] == "speed":
                pl_speed = np.append(pl_speed, float(means[i, 1]))

        pl_data = self._cut_relevant_results(data=pl_data, data_type="sum", cut_from_bas=4, cut_from_add=2)
        pl_speed = self._cut_relevant_results(data=pl_speed, data_type="avr", cut_from_bas=4, cut_from_add=2)

        # bar = pl_data.astype(float)
        bar = pl_data

        r = np.arange(len(bar))

        pl_data1 = np.around(pl_data.astype(float)).astype(int)

        plt.bar(r, bar, width=bar_width, color='yellow', edgecolor='black', label='20 m/s')

        for i in range(len(r)):
            plt.text(x=r[i]-0.2, y=pl_data1[i]+50, s=f"v={int(pl_speed[i])}", color='black', rotation=90)
        plt.xticks([r for r in range(len(bar))], x_label)
        plt.ylabel("Value")
        plt.xlabel("Case")
        plt.title(f"{data} - 100")
        plt.legend()

        plt.show()

        pl_speed, pl_data = zip(*sorted(zip(pl_speed, pl_data)))

        plt.plot(pl_speed, pl_data)
        plt.ylabel(f"{data} {self.dimensions[data]}")
        plt.xlabel("Speed [m/s]")
        plt.title(f"{data} - 100")

        plt.show()


def main(mode):
    """
    Main function of plotter.py. Calls Plotter class and its functions according to it's mode.
    Generate mode: Create plots for every case respectively.
    Plot mode: Create plots which shows difference between test cases.
    """
    # plotter = Plotter()
    # plotter.read_file(
    #     filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/highway_case_4_emission.csv")
    # plotter.fix_zero_emissions()
    # plotter.calculate_average()
    # plotter.plot_results(
    #     filename=f"/home/akos/workspace/Thesis/thesis/data/20200217_first_results/plots/Plot_Case_000_")
    folder_name = "20200320"
    if mode == "generate":
        for a in range(1, 37):
            plotter = Plotter(folder_name=folder_name)
            plotter.read_file(filename=f"/home/akos/workspace/emission_results/"
                                       f"{folder_name}/highway_case_{a}_emission.csv")
            plotter.fix_zero_emissions()
            plotter.calculate_average()
            plotter.plot_results(filename=f"/home/akos/workspace/emission_results/"
                                          f"{folder_name}/plots/Plot_Case_{a}_")

    elif mode == "plot":
        filename = f"/home/akos/workspace/emission_results/{folder_name}/results_all.csv"
        save_path = f"/home/akos/workspace/emission_results/{folder_name}/plots/results/"
        plotter = Plotter(folder_name=folder_name)
        # plotter.plot_cases(filename=filename, cases=cases, data="fuel")
        data_avr = [["fuel", "[ml/s]", "consumption"],
                    ["CO", "[g/s]", "emission"],
                    ["CO2", "[g/s]", "emission"],
                    ["NOx", "[g/s]", "emission"],
                    ["HC", "[g/s]", "emission"],
                    ["PMx", "[g/s]", "emission"]]
        data_sum = [["fuel", "[ml]", "consumption"],
                    ["CO", "[g]", "emission"],
                    ["CO2", "[g]", "emission"],
                    ["NOx", "[g]", "emission"],
                    ["HC", "[g]", "emission"],
                    ["PMx", "[g]", "emission"]]
        data_types = ["Sum", "Avr"]
        start_indexes = np.array([0, 12, 24])
        for start_index in start_indexes:
            for data_type in data_types:
                for data_name_dim_exp in (data_avr if data_type == "Avr" else data_sum):
                    plotter.plot_difference(filename=filename,
                                            start_index=start_index,
                                            data_type=data_type,
                                            data_name_dim_exp=data_name_dim_exp,
                                            save_path=save_path)
                    plotter.plot_values_and_speed(filename=filename,
                                                  start_index=start_index,
                                                  data_type=data_type,
                                                  data_name_dim_exp=data_name_dim_exp,
                                                  save_path=save_path)
        # plotter.plot_same_flow_different_cases(filename=filename, data="fuel")


if __name__ == "__main__":

    main(mode="plot")

    print("End")
