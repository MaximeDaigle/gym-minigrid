import argparse
import time
import datetime
import torch
import tensorboardX
import gym
import gym_minigrid
import sys

import utils
from train_mgmt import training_management
from supervised_model import CNN_LSTM


## General parameters
parser = argparse.ArgumentParser()

parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log_interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save_interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--updates", type=int, default=10000,
                    help="number of updates of training (default: 10,000)")
parser.add_argument("--visualize", action="store_true", default=False,
                    help="Show last frame of the last sample for every log interval")
parser.add_argument("--batch_size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")

args = parser.parse_args()

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments
utils.seed(args.seed)
env = gym.make(args.env)
env.seed(args.seed)

# Load training status
try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"update": 0}
txt_logger.info("Training status loaded\n")

# Load model
model = CNN_LSTM(obs_shape=env.observation_space.spaces["image"].shape,  nb_class=env.width * env.height)
if "model_state" in status:
    model.load_state_dict(status["model_state"])
model.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(model))

mgmt = training_management(env, model, device, args.lr, args.batch_size)

if "optimizer_state" in status:
    mgmt.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Training
update = status["update"]
start_time = time.time()
losses = []
accuracy = torch.tensor([]).to(device)
over = False
while update < args.updates and not over:

    # Train
    update_start_time = time.time()
    images, label, seq_lens = mgmt.collect_episode()
    loss, correct = mgmt.update_parameters(images, label, seq_lens)

    # Log
    losses.append(loss)
    accuracy = torch.cat((accuracy, correct), 0)
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        if args.visualize:
            # Visualize last frame of last sample
            from gym_minigrid.window import Window
            window = Window('gym_minigrid - ' + args.env)
            images = images.transpose(1,2)
            images = images.transpose(2,3)
            print(images[-1].shape)
            print(label)
            window.show_img(images[-1])
            input()
            window.close()

        duration = int(time.time() - start_time)

        header = ["Update", "Time", "Loss", "Accuracy"]
        acc = torch.mean(accuracy)
        data = [update, duration, sum(losses) / len(losses), acc]
        losses = []
        over = (acc >= 0.9999)
        accuracy = torch.tensor([]).to(device)

        txt_logger.info(
            "U {} | T {} | L {:.3f} | A {:.4f}"
            .format(*data))

        if status["update"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, update)

    # Save status
    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"update": update,
                  "model_state": model.state_dict(), "optimizer_state": mgmt.optimizer.state_dict()}
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
