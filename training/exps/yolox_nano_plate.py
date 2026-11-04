# Custom YOLOX-Nano configuration for Vietnamese license plate detection
import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Architecture parameters for YOLOX-Nano
        self.depth = 0.33
        self.width = 0.25
        self.scale = (0.5, 1.5)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Single class: license plate
        self.num_classes = 1

        # Configurable epochs (defaults to 150)
        self.max_epoch = int(os.environ.get("YOLOX_EPOCHS", "150"))
        self.no_aug_epochs = max(10, int(self.max_epoch * 0.10))
        self.data_num_workers = 4
        self.eval_interval = 5
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.basic_lr_per_img = 0.01 / 64.0

        # Dataset paths (relative to training directory or absolute)
        self.data_dir = os.environ.get("YOLOX_DATA_DIR", "dataset_coco")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

