import torch
from gymnasium import Env, spaces
from torch.utils.data import DataLoader
import minari
import numpy as np



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
        "id": torch.Tensor([ep.id for ep in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(
                [
                    [*ag, *dg, *obs] for ag, dg, obs in zip(ep.observations['achieved_goal'], ep.observations['desired_goal'], ep.observations['observation'])
                ]
                ) for ep in batch], batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(ep.actions) for ep in batch], batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(ep.rewards) for ep in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(ep.truncations) for ep in batch], batch_first=True
        ),
    }


if __name__ == "__main__":
    data_path = "PandaSlide/test-v0"
    batch_size = 2  # How many episodes per batch
    dataloader, env = load_dataset(data_path, batch_size)
    
    obs_space, act_space = env.observation_space, env.action_space
    assert isinstance(obs_space, spaces.Dict)
    assert isinstance(act_space, spaces.Box)
    print(np.prod(obs_space['achieved_goal'].shape) + np.prod(obs_space['desired_goal'].shape) + np.prod(obs_space['observation'].shape)) # type: ignore
    print(np.prod(act_space.shape))

