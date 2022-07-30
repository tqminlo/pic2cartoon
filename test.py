import numpy as np
import cv2
from models.generator import Generator


def gen_pic2cartoon(pic_path, cartoon_path, size=256):
    model = Generator(size).gmodel()
    model.load_weights("weights/first_save.h5")
    x = cv2.imread(pic_path)
    x = cv2.resize(x, (size, size))
    x = [x]
    x = np.array(x)
    x = x / 255.

    out = model.predict(x)
    out = out[0] * 255
    cv2.imwrite(cartoon_path, out)

    return cartoon_path



if __name__ == "__main__":
    pic_path = "test_result/inp/00000063.jpg"
    cartoon_path = "test_result/inp/00000063_cartoon.jpg"
    gen_pic2cartoon(pic_path, cartoon_path)


