import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.marker'] = 'o'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# python3 plot.py ./execution_time_k1.txt 16 1024 kernel1

def plot():
    if len(sys.argv) != 5:
        print("Usage: python plot.py <file.txt> <number> <base> <image_name>")
        return
    n = int(sys.argv[2])
    base = int(sys.argv[3])
    shapelist = np.arange(base, base+(n*base), base)
    ylist = np.arange(5000, 2500+25000, 2500)
    # print(shapelist)

    file_path = sys.argv[1]
    with open(file_path, "r") as f:
        time = np.loadtxt(f).transpose()
    GFlops = (2*shapelist-1)*(shapelist**2)/time / 1e9
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot()
    ax1.plot(shapelist, GFlops[0], label=f"{sys.argv[4]}")
    ax1.plot(shapelist, GFlops[1], label="cublas", color='red')
    ax1.axes.set_xticks(shapelist)
    ax1.axes.set_yticks(ylist)
    ax1.legend()
    ax1.set_title("Average GFLOP/s for 50 cycles")
    ax1.set_xlabel("matrix dimension")
    ax1.set_ylabel("GFLOP/s")

    fig1.autofmt_xdate()
    plt.savefig(f"{sys.argv[4]}.png")
    plt.show()

if __name__ == "__main__":
    plot()