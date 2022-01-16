import pandas as pd
import numpy as np
import os, re, time
from scipy import interpolate
import matplotlib.pyplot as plt

def read_txt(file):
    data = []
    f = open(file)
    for id, line in enumerate(f.readlines()):
        if id == 0:
            pass
        else:
            s = re.sub(' +', ' ', line.strip()).split(" ")
            data.append(list(map(float, s)))
    return pd.DataFrame(data)

def processing(datas, max_time):
    time = datas[:, 0]
    conc = datas[:, 1:3]
    sensors = datas[:, 3:]
    new_time = np.linspace(0, max_time, max_time*100+1)[:,np.newaxis]
    new_datas = np.linspace(0, max_time, max_time*100+1)[:,np.newaxis]
    for i in range(2):
        new_conc = conc_processing(new_time, time, conc[:, i])
        new_datas  = np.column_stack((new_datas, new_conc))
    for i in range(16):
        F = interpolate.interp1d(time, sensors[:, i], kind="linear")
        new_sensor = F(new_time)
        new_datas  = np.column_stack((new_datas, new_sensor))
    return new_datas

def conc_processing(new_time, time, conc):
    new_conc = np.array([])
    i = 0
    for step in new_time[:, 0]:
        if round(time[i], 2) == round(step, 2):
            new_conc = np.append(new_conc, conc[i])
            i = i + 1
        elif round(time[i], 2) > round(step, 2):
            new_conc = np.append(new_conc, conc[i])
            # print(i, round(step, 2), round(time[i], 2))
    return new_conc[:,np.newaxis]

def plot(new_datas, name):
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(name)
    ax[0].plot(new_datas[:, 0], new_datas[:, 1], label="CO", c="r")
    ax[0].plot(new_datas[:, 0], new_datas[:, 2], label="Ethylene",c="b")
    ax[0].set_title("Gas Concentration")
    
    ax[1].plot(new_datas[:, 0], new_datas[:, 3], label="TGS2602", c="red")
    ax[1].plot(new_datas[:, 0], new_datas[:, 5], label="TGS2600", c="orange")
    ax[1].plot(new_datas[:, 0], new_datas[:, 7], label="TGS2610", c="gold")
    ax[1].plot(new_datas[:, 0], new_datas[:, 9], label="TGS2620", c="green")
    ax[1].set_title("Sensor Response")
    plt.savefig("{}.jpg".format(name), dpi=300)

if __name__ == "__main__":
    begin = time.time()
    for file in os.listdir("./dataset"):
        if "ethylene_CO.txt" in file:
            f = read_txt(os.path.join("./dataset", file))
            f.drop_duplicates(subset=0, keep='first', inplace=True)
            f = f.values
            new_datas = processing(f, 42087)
            plot(new_datas, file.split(".")[0])
            new_datas = pd.DataFrame(new_datas)
            new_datas.to_csv("{}.csv".format(file.split(".")[0]), header=None, index=False)
            print("time: %.2f s", (time.time() - begin) / 3600)

        if "ethylene_methane.txt" in file:
            f = read_txt(os.path.join("./dataset", file))
            f.drop_duplicates(subset=0, keep='first', inplace=True)
            f = f.values
            new_datas = processing(f, 41790)
            plot(new_datas, file.split(".")[0])
            new_datas = pd.DataFrame(new_datas)
            new_datas.to_csv("{}.csv".format(file.split(".")[0]), header=None, index=False)
            print("time: %.2f h", (time.time() - begin) / 3600)
    print("All time: %.2f s", (time.time() - begin) / 3600)