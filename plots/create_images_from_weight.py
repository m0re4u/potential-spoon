import sys
import matplotlib.pyplot as plt


def main():
    with open('../build/network.log') as f:
    # with open('../weights/10000run2.log') as f:
        imgmatrix = []
        for line in f:
            la = [
                float(x) for x in line.strip(" ").strip("\n").split(",")[0:-1]
            ]
            if len(la) != 28:
                sys.exit(1)
            imgmatrix.append(la)
            if len(imgmatrix) == 28:
                plt.imshow(imgmatrix, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.show()
                imgmatrix = []


if __name__ == '__main__':
    main()
