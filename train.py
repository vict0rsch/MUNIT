"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from comet_ml import Experiment

from utils import (
    get_all_data_loaders,
    prepare_sub_folder,
    write_html,
    write_loss,
    get_config,
    write_2images,
    Timer,
)
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer, DoubleMUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/edges2handbags_folder.yaml",
    help="Path to the config file.",
)
parser.add_argument("--output_path", type=str, default=".", help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--trainer", type=str, default="MUNIT", help="MUNIT|UNIT")
parser.add_argument("--seed", type=int, default=None, help="Torch and numpy seeds")
opts, unknownargs = parser.parse_known_args()

comet_exp = Experiment(project_name="munit", workspace="vict0rsch")

cudnn.benchmark = False
if opts.seed is not None:
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True

# Load experiment setting
config = get_config(opts.config)
if unknownargs:
    for u in unknownargs:
        try:
            k, v = u.split("=")
            k = k.strip().replace("-", "")
            v = v.strip()
            if v.replace(".", "").isdigit():
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            if k in config:
                print("Overwriting {:20} {:30} -> {:}".format(k, config[k], v))
                config[k] = v
        except Exception as e:
            print(e)
            print("Ignoring argument", u)

    for o in dir(opts):
        if not o.startswith("_"):
            if o in config:
                print(
                    "Overwriting {:20} {:30} -> {:}".format(
                        o, config[k], getattr(opts, o)
                    )
                )
                config[o] = getattr(opts, o)

comet_exp.log_asset(opts.config)
max_iter = config["max_iter"]
display_size = config["display_size"]
config["vgg_model_path"] = opts.output_path

comet_exp.log_parameters(config)

# Setup model and data loader
if opts.trainer == "MUNIT":
    trainer = MUNIT_Trainer(config, comet_exp)
elif opts.trainer == "UNIT":
    trainer = UNIT_Trainer(config)
elif opts.trainer == "DoubleMUNIT":
    trainer = DoubleMUNIT_Trainer(config, comet_exp)
else:
    sys.exit("Only support MUNIT|UNIT|DOubleMUNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(
    config
)
train_display_images_a = torch.stack(
    [train_loader_a.dataset[i] for i in range(display_size)]
).cuda()
train_display_images_b = torch.stack(
    [train_loader_b.dataset[i] for i in range(display_size)]
).cuda()
test_display_images_a = torch.stack(
    [test_loader_a.dataset[i] for i in range(display_size)]
).cuda()
test_display_images_b = torch.stack(
    [test_loader_b.dataset[i] for i in range(display_size)]
).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(
    os.path.join(opts.output_path + "/logs", model_name)
)
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(
    opts.config, os.path.join(output_directory, "config.yaml")
)  # copy config file to output folder

# Start training
iterations = (
    trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
)

while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config["log_iter"] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer, comet_exp=comet_exp)

        # Write images
        if (iterations + 1) % config["image_save_iter"] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(
                    test_display_images_a, test_display_images_b
                )
                train_image_outputs = trainer.sample(
                    train_display_images_a, train_display_images_b
                )
            write_2images(
                test_image_outputs,
                display_size,
                image_directory,
                "test_%08d" % (iterations + 1),
                comet_exp=comet_exp,
            )
            write_2images(
                train_image_outputs,
                display_size,
                image_directory,
                "train_%08d" % (iterations + 1),
                comet_exp=comet_exp,
            )
            # HTML
            write_html(
                output_directory + "/index.html",
                iterations + 1,
                config["image_save_iter"],
                "images",
                comet_exp=comet_exp,
            )

        if (iterations + 1) % config["image_display_iter"] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(
                    train_display_images_a, train_display_images_b
                )
            write_2images(image_outputs, display_size, image_directory, "train_current")

        # Save network weights
        if (iterations + 1) % config["snapshot_save_iter"] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit("Finish training")
