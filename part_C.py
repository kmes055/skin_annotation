import cv2, numpy as np
from os import listdir
from cmath import cos, sin
import gc

annotation_path = 'FDDB-folds/'
img_path = 'originalPics/'
files = listdir(annotation_path)
files = files[0::2] if len(files[0]) > 20 else files[1::2]


def parse_img(shape, row):
    row = [float(r) for r in row]
    long, short, angle, x, y, s = row
    focus = np.sqrt(long**2 - short**2)

    focus_a = [x + cos(angle)*focus, y + sin(angle)*focus]
    focus_b = [x - cos(angle)*focus, y - sin(angle)*focus]

    def dist(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def checkpoint(p):
        return dist(p, focus_a) + dist(p, focus_b) < 2*long

    mask = [[0 if checkpoint([x, y]) else 1 for x in range(shape[1])] for y in range(shape[0])]
    return np.array(mask, dtype=np.uint64)


def create_data():
    photo_data = []
    images = []
    cat_img = {}

    for file in files[:1]:
        with open(annotation_path + file, 'r') as e:
            lines = e.readlines()
        image = []
        images = []
        for line in lines:
            if len(line.split('/')) > 2:  # if image path
                images.append(image) if image != [] else None
                image = [line[:-1]]
            elif len(line.split()) == 6:
                image.append(line.split())

        images_mask = {}

        pnt = 0
        for imginfo in images[-10:]:
            name, anno = imginfo[0], imginfo[1:]
            img = cv2.imread(img_path + name + '.jpg')
            images_mask[name] = np.ones(img.shape[:2])

            for an in anno:
                images_mask[name] = np.logical_and(images_mask[name], parse_img(img.shape, an))
            images_mask[name] = images_mask[name].astype(np.int) + 1

            label = np.dstack((img, images_mask[name]))
            photo_data.extend(label.reshape(-1, 4))
            pnt += 2

            # For imshow
            backup = img.copy()
            img[images_mask[name] == 2] = [0, 0, 0]
            cat_img[name] = np.hstack((backup, img))

        ret = np.array(photo_data)
        ret = np.unique(ret, axis=0)
        photo_data = [[str(x) for x in row] for row in photo_data]
        photo_data = ['\t'.join(row) + '\n' for row in photo_data]
        photo_data = list(set(photo_data))

    with open('PicData.txt', 'w+') as f:
        f.writelines(photo_data)

    return ret, cat_img


def get_data():  # Not used.
    with open('PicData.txt', 'r') as f:
        lines = f.readlines()
        if len(lines) < 10:
            return create_data(), None
        else:
            return np.array([[int(n) for n in line.split()] for line in lines]), None


def predict_image(model, img_dic):
    for i, imgpath in enumerate(img_dic.keys()):
        img = cv2.imread(img_path + imgpath + '.jpg')
        predict = model.predict(img.reshape(-1, 3)).reshape(img.shape[:2])
        img[predict == 1] = [255, 255, 255]
        img[predict == 2] = [0, 0, 0]

        result_img = np.hstack((img_dic[imgpath], img))
        cv2.imwrite('results\\result %d.jpg' % (i + 1), result_img)