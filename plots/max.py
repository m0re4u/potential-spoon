import sys
import numpy as np
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(str(sys.argv[1])) as f:
        num = [float(x.strip('\n'))*1000 for x in f]
        print("Average {}".format(np.average(num)))

    bins = np.linspace(math.ceil(min(num)),
                       math.floor(max(num)),
                       num=20)

    print(bins)
    print(np.array(num))
    plt.xlim([min(num)-5, max(num)+5])

    plt.hist(num, bins=bins, alpha=0.5)
    plt.xlabel('variable X (20 evenly spaced bins)')
    plt.ylabel('count')

    plt.show()
