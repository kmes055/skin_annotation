import numpy as np


def get_data():
    with open('Skin_NonSkin.txt', 'r') as skin:
        text_data = skin.readlines()
        text_data = np.array([[int(n) for n in line.split()] for line in text_data])
    return text_data
