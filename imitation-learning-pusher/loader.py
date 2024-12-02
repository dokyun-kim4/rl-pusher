import torch
from gymnasium import Env
from torch.utils.data import DataLoader
import minari



def load_dataset(path: str, batch_size: int) -> tuple[DataLoader, Env]:
    minari_dataset = minari.load_dataset(path)
    env = minari_dataset.recover_environment()
    print("----------Successfully loaded environment---------")
    print("Observation space:", minari_dataset.observation_space)
    print("Action space:", minari_dataset.action_space)
    print("Total episodes:", minari_dataset.total_episodes)
    print("Total steps:", minari_dataset.total_steps)
    print("--------------------------------------------------")
    dataloader = DataLoader(minari_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # type: ignore

    return dataloader, env


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch], batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch], batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


if __name__ == "__main__":
    data_path = "pusher/expert-v0"
    batch_size = 2  # How many episodes per batch
    dataloader, env = load_dataset(data_path, batch_size)

    for batch in dataloader:
        print(batch)
        break

