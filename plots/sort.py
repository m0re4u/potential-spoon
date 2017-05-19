

if __name__ == '__main__':
    with open('../build/network.log') as f:
        ls = sorted([float(line.split(',')[1]) for line in f])
        print(*ls, sep="\n")
