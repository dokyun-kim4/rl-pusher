import torch
from gymnasium import Env
from torch.utils.data import DataLoader
import minari



def load_dataset(path: str, batch_size: int) -> tuple[DataLoader, Env]:
    """
    Loads a dataset from a Minari file and recovers its environment.

    Args:
        path (str): The file path to the Minari dataset.
        batch_size (int): The number of samples per batch for the DataLoader.

    Returns:
        tuple[DataLoader, Env]: A tuple containing:
            - dataloader (DataLoader): A DataLoader for the dataset with the specified batch size.
            - env (Env): The recovered environment from the dataset.
    """
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
    """
    Collates a batch of data samples into padded tensors for training.

    Args:
        batch (list): A list of samples, where each sample is an object containing:
            - id (int): The sample ID.
            - observations (list): The sequence of observations.
            - actions (list): The sequence of actions.
            - rewards (list): The sequence of rewards.
            - truncations (list): The sequence of truncations.

    Returns:
        dict: A dictionary with the following keys:
            - "id" (torch.Tensor): A tensor of sample IDs.
            - "observations" (torch.Tensor): Padded tensor of observations (batch-first).
            - "actions" (torch.Tensor): Padded tensor of actions (batch-first).
            - "rewards" (torch.Tensor): Padded tensor of rewards (batch-first).
            - "truncations" (torch.Tensor): Padded tensor of truncations (batch-first).
    """
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
    data_path = "pusher/expert-v1"
    batch_size = 100  # How many episodes per batch
    dataloader, env = load_dataset(data_path, batch_size)

    for batch in dataloader:
        print(batch['observations'].shape)
        print(batch['actions'].shape)
        

