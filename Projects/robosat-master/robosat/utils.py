import matplotlib.pyplot as plt


def plot(out, history):
    plt.figure()

    n = max(map(len, history.values()))
    plt.xticks(range(n), [v + 1 for v in range(n)])

    plt.grid()

    for values in history.values():
        plt.plot(values)

    plt.xlabel('epoch')
    plt.legend(list(history))

    plt.savefig(out, format='png')
    plt.close()
