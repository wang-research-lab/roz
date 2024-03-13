import numpy as np
from torchvision.datasets import CIFAR10
import pickle
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


def put_text(image, text, count):
    def check_point(target_x, target_y):
        def check_single_point(x, y):
            for point in points_li:
                if x > point[0] and x < point[0] + width and y < point[1] and y > point[1] - height:
                    return False
            return True
        return check_single_point(target_x, target_y) & check_single_point(target_x+width, target_y) \
        & check_single_point(target_x, target_y-height) & check_single_point(target_x+width, target_y-height)
    height = 2
    width = len(text) / 6 * 5
    points_li = []
    for i in range(4):
        while True:
            x = np.random.randint(0, 24)
            y = np.random.randint(1, 32)
            if check_point(x,y):
                break

        points_li.append((x,y))

    plt.imshow(image)
    for point in points_li:
        plt.text(point[0], point[1], text, fontdict=font)


    plt.savefig(os.path.join(base_path, str(count) + ".JPEG"), pad_inches=0, dpi=150)
    plt.cla()







def build_synsets_dict():
    synsets_li = ['airplane','automobile', 'bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    reverse_dict = {label:i for i, label in enumerate(synsets_li)}
    return synsets_li, reverse_dict


def build_dir():
    base_path = './CIFARtypographic'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    return base_path



if __name__ == '__main__':
    base_path = build_dir()
    record = []
    count_num = 0
    synsets_dict, reverse_dict = build_synsets_dict()

    font = {'color': 'white',
            'size': 20,
            'family': 'Times New Roman', }
    dataset = CIFAR10(root=CIFARPATH, train=False)
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        mis_label = label
        while mis_label == label:
            mis_label = np.random.randint(10)
        put_text(img, synsets_dict[mis_label], i)
        record.append((label, mis_label))


    with open('cifar10_record.pickle', 'wb') as f:
        pickle.dump(record, f)

