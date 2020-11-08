import numpy as np
import torch


class training_management():

    def __init__(self, env, model, device=None, lr=0.001, batch_size=256):
        self.env = env
        self.model = model
        self.device = device
        self.lr = lr
        self.model.to(self.device)
        self.model.train()
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def collect_episode(self):
        """
        Collects frames and label for a batch
        """
        batch = []
        batch_label = []
        for i in range(self.batch_size):
            frames = [self.env.reset()]
            done = False
            while not done: # complete an episode
                obs, label, done, _ = self.env.step()
                frames.append(obs)
            frames = np.stack(frames, axis=0) # frames of one episode, shape: (nb_frames, y, x, channel)
            frames = torch.tensor(frames, device=self.device, dtype=torch.float)
            frames = frames.transpose(2, 3)
            frames = frames.transpose(1, 2)  # shape: (nb_frames, channel, y, x)
            frames = frames / 255 # scale pixel values to [0,1]
            batch.append(frames)
            batch_label.append(label)

        # remove timesteps dimension because feed into cnn first and cnn processes each image individually
        seq_lens = [] # count number of frames for each sample to reassemble them later
        for i in range(len(batch)): # get the different nb_frames (i.e. seq_len) for each sample
            seq_lens.append(batch[i].shape[0]) # x[i] shape: (nb_frames, channel, y, x)
        batch = torch.cat(batch, dim=0)
        batch_label = torch.tensor(batch_label, device=self.device, dtype=torch.long)
        return batch, batch_label, seq_lens


    def update_parameters(self, input, label, seq_lens):
        self.optimizer.zero_grad()
        output = self.model(input, seq_lens)
        correct = (torch.argmax(output, dim=1) == label)
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss, correct
