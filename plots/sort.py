import sys

if __name__ == '__main__':
    with open(str(sys.argv[1])) as f:
        ls = sorted([int(line.split(",")[1]) for line in f])
        print("\n".join(list(map(str, ls))))
