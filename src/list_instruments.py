import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint

d = defaultdict(lambda: 0)

PATH = "/Users/nitkonitkic/Documents/LumenDataSci/data_out"


def main():

    for x in tqdm(os.listdir(PATH), total=len(list(os.listdir(PATH)))):

        a = np.load(PATH + "/" + x, allow_pickle=True)

        for y in a["arr_0"]:
            for z in y[0].split("_")[1:]:
                d[z] += 1

        pprint(d)


if __name__ == "__main__":
    main()
