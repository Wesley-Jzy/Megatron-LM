import torch
import numpy as np
import random
import string


def get_data(micro_batch_size, seq_len, vocab_size):
    t = torch.randint(0, vocab_size, (seq_len,))
    return t

class RandomTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_num, micro_batch_size, seq_len):
        super().__init__()
        self.data = []
        self.data_num = data_num
        for i in range(data_num):
            self.data.append(get_data(micro_batch_size, seq_len, 20000))
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len

    def __getitem__(self, index):
        full_seq = self.data[index]
        return {"text": full_seq}

    def __len__(self):
        return len(self.data)


def build_data_from_random(
    micro_batch_size: int = 12,
    seq_len: int = 1024,
):
    train_dataset = RandomTextDataset(10000, micro_batch_size, seq_len)
    valid_dataset = RandomTextDataset(1000, micro_batch_size, seq_len)
    test_dataset = RandomTextDataset(1000, micro_batch_size, seq_len)

    return train_dataset, valid_dataset, test_dataset