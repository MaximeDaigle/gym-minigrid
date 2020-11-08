import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, obs_shape, nb_class, lstm_hidden_size=128):
        super(CNN_LSTM, self).__init__()

        kernel_size = (5, 5)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        width = obs_shape[1] # input image width before cnn
        for i in range(3):
            width = ((width - kernel_size[0]) + 1) // 2  # compute image width after cnn

        lstm_input_size = width*width*64
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True)

        self.linear = nn.Linear(lstm_hidden_size, nb_class)

    def forward(self, x, seq_lens):
        x = self.cnn(x)

        x = torch.flatten(x, start_dim=1) # (batch_size * nb_total_images, nb_channel_output * ouput_heigh * output_width)
        x = torch.split(x, seq_lens) # list of tensors which is of length batch_size and each tensor (seq_len, input_size)

        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, h_c) = self.lstm(x)
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # shape (batch_size, max_seq_length, )
        last_indices = lens_unpacked - 1
        out = seq_unpacked[[i for i in range(seq_unpacked.shape[0])],last_indices,:] # take last hidden state for each samples

        out = self.linear(out) # feed the last hidden state into a linear projection
        return out