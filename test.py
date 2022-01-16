import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = pd.read_csv("./dataset/ethylene_CO_1Hz_smooth.csv", header=None).values[:, 3:]
    length = len(f)
    time = np.arange(1, length+1, 1)[:, np.newaxis]
    plt.figure(1)
    plt.title("Fig.1 sensor error")
    plt.plot(time, f[:, 1], color="r", label="abnormal")
    plt.plot(time, f[:, 5], color="b", label="normal")
    plt.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Conducticity(S)")
    # plt.ylim([0, 5000])

    length = len(f[10:, :])
    time = np.arange(1, length+1, 1)[:, np.newaxis]
    plt.figure(2)
    plt.title("Fig.2 sensor disturbution")
    plt.plot(time, f[10:, 0], color="salmon", label="sensor 1-1")
    plt.plot(time, f[10:, 4], color="orange", label="sensor 1-2")
    plt.plot(time, f[10:, 8], color="gold", label="sensor 1-3")
    plt.plot(time, f[10:, 12], color="green", label="sensor 1-4")
    plt.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Conducticity(S)")
    plt.show()
