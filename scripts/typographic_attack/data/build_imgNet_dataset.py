import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm
import scipy.io as sio

count = 0
def put_text(image, text):
    def check_point(target_x, target_y):
        def check_single_point(x, y):
            for point in points_li:
                if x > point[0] and x < point[0] + width and y < point[1] and y > point[1] - height:
                    return False
            return True
        return check_single_point(target_x, target_y) & check_single_point(target_x+width, target_y) \
        & check_single_point(target_x, target_y-height) & check_single_point(target_x+width, target_y-height)
    m = image.shape[1]
    n = image.shape[0]
    global count
    (width, height), baseLine = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=2)
    if m-width <= 0 or height >= n:
        count += 1
        return False
    points_li = []
    for i in range(8):
        while True:
            x = np.random.randint(0, m - width)
            y = np.random.randint(0 + height, n)
            if check_point(x,y):
                break

        points_li.append((x,y))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return True




def get_mis_text(img, label):
    def check(text):
        (width, height), baseLine = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=2)
        if m - width <= 0 or height >= n:
           return True
        else:
            return False

    m = img.shape[1]
    n = img.shape[0]

    count = 0
    while True:
        cur_num = np.random.randint(len(synsets_dict))
        if cur_num != label and cur_num in valid_labels_set and check(synsets_dict[reverse_dict[cur_num]][1]):
            break
        count += 1
        if count >= 100:
            if cur_num != label and cur_num in valid_labels_set:
                return cur_num


    return cur_num


def build_synsets_dict():
    synsets_info_path = './ILSVRC2012_devkit_t12/data/meta.mat'
    raw_data = sio.loadmat(synsets_info_path)['synsets']

    synsets_list = []
    reverse_dict = {}

    for i, synset in enumerate(raw_data):
        if i == 1000:
            break
        orig_synset = synset[0]

        synset_id = orig_synset[1].item()
        synset_label = orig_synset[2].item()
        gloss = orig_synset[3].item()
        synsets_list.append((synset_id, synset_label, gloss))

    sorted_list = sorted(synsets_list, key=lambda item: item[0])

    synsets_dict = {}
    for i, (synset_id, synset_label, gloss) in enumerate(sorted_list):
        synsets_dict[synset_id] = (i, synset_label, gloss)
        reverse_dict[i] = synset_id

    return synsets_dict, reverse_dict


def get_valid_label_set():
    valid_set = set()
    for synset_id, (label_id, synset_label, _) in synsets_dict.items():
        if ',' not in synset_label and ' ' not in synset_label:
            valid_set.add(label_id)
    print(len(valid_set))
    return valid_set


def build_dir():
    synsets_dict, reverse_dict = build_synsets_dict()
    base_path = SavePATH
    for synset_id, _ in synsets_dict.items():
        construct_path = os.path.join(base_path, synset_id)
        os.mkdir(construct_path)



if __name__ == '__main__':
    build_dir()
    record = {}
    count_num = 0


    synsets_dict, reverse_dict = build_synsets_dict()
    valid_labels_set = get_valid_label_set()

    root_path = ImageNetPATH
    synset_label_li = os.listdir(root_path)
    for synset_path in tqdm(synset_label_li):
        golden_id = synsets_dict[synset_path][0]
        golden_label = synsets_dict[synset_path][1]
        full_synset_path = os.path.join(root_path, synset_path)
        image_path_li = os.listdir(full_synset_path)
        for image_path in image_path_li:
            full_image_path = os.path.join(full_synset_path, image_path)
            image = cv2.imread(full_image_path)

            cur_num = get_mis_text(image, golden_id)

            mis_label = synsets_dict[reverse_dict[cur_num]][1]
            flag = put_text(image, mis_label)
            if flag:
                write_image_path = os.path.join(SavePATH, synset_path, image_path)
                cv2.imwrite(write_image_path, image)
                record[image_path] = cur_num


    with open('imgNet_record.pickle', 'wb') as f:
        pickle.dump(record, file=f)

    with open('valid_label_set.pickle', 'wb') as f:
        pickle.dump(valid_labels_set, file=f)


    print(count)


