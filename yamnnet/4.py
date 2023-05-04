import json
from sklearn.metrics import hamming_loss


def main():
    for_scoring = json.load(open('for_scoring.json'))

    a1, a2 = [], []

    l = []

    for x in for_scoring:
        # print(x[0])
        a1 = [1 if y in x[1] else 0 for y in range(11)]
        a2 = [1 if y in x[2] else 0 for y in range(11)]

        l.append(1 - hamming_loss(a1, a2))

    print(sum(l) / len(l))

    # print(hamming_loss(a1, a2))


if __name__ == '__main__':
    main()
