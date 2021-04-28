import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util
from pathlib import Path


"""
Single-Scale Unaligned Dataset
Use with CUTGAN Implementation
"""
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.no_flip = True
    dataset = create_dataset(opt)
    model = create_model(opt)
    A_size = 700

    for i, data in enumerate(dataset):
        if i==0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
        if i>=A_size:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        for label, image in visuals.items():
            if label == "fake_B":
                image_numpy = util.tensor2im(image)
                file_path = model.get_image_paths()[0]
                useless, tail = os.path.split(file_path)
                file_name = Path(str(i) + "_" + str(tail))
                img_path = os.path.join("datasets/generated/", file_name)
                print(i, img_path)
                util.save_image(image_numpy, img_path)