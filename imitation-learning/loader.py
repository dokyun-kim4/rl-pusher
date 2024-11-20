import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import minari



def load_dataset(path: str, batch_size: int) -> DataLoader:
    minari_dataset = minari.load_dataset(path)
    print("----------Successfully loaded environment---------")
    print("Observation space:", minari_dataset.observation_space)
    print("Action space:", minari_dataset.action_space)
    print("Total episodes:", minari_dataset.total_episodes)
    print("Total steps:", minari_dataset.total_steps)
    print("--------------------------------------------------")
    dataloader = DataLoader(minari_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) # type: ignore

    return dataloader

def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }


if __name__ == "__main__":
    data_path = "pusher/expert-v0"
    batch_size = 1 # How many episodes per batch
    dataloader = load_dataset(data_path, batch_size)
        