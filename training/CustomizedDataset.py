import pickle
import torch
from torch.utils.data import Dataset

class CustomizedDataset(Dataset):
    def __init__(self, metadata):
        super(CustomizedDataset, self).__init__()
        self.data = pickle.load(open(metadata, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # text = torch.tensor(item['text_embed'], dtype=torch.float32).clone().detach()
        # audio = torch.tensor(item['audio_embed'], dtype=torch.float32).clone().detach()
        # label = torch.tensor(item['label'], dtype=torch.long)
        text = item['text_embed'].clone().detach().float() if torch.is_tensor(item['text_embed']) else torch.tensor(item['text_embed'], dtype=torch.float32)
        audio = item['audio_embed'].clone().detach().float() if torch.is_tensor(item['audio_embed']) else torch.tensor(item['audio_embed'], dtype=torch.float32)
        label = item['label'].clone().detach().long() if torch.is_tensor(item['label']) else torch.tensor(item['label'], dtype=torch.long)
        return text, audio, label
