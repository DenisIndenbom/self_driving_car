import keyboard as kb
import torch
from numpy.random import choice

from base import Agent, State, Action
from car import CarAction

__all__ = ['PlayerAgent', 'AIAgent', 'Sample']


class PlayerAgent(Agent):
    """ Default car agent for debugging """

    def step(self, state: State) -> Action:
        up = kb.is_pressed('up')
        down = kb.is_pressed('down')
        right = kb.is_pressed('right')
        left = kb.is_pressed('left')

        direction = 0
        if up and not down:
            direction = 1
        elif down and not up:
            direction = 2

        turn = 0
        if right and not left:
            turn = 1
        elif left and not right:
            turn = 2

        return CarAction((direction * 3) + turn)

    def observe(self, state: State, action: Action, new_state: State, reward: int | float):
        pass

    def update_policy(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def reset(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


# TODO: Delete the shit below and make a new shit

class NN(torch.nn.Module):
    def __init__(self, n_actions: int, input_size: int, hidden_size: int):
        super().__init__()

        self.act = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.fc_input = torch.nn.Linear(n_actions + input_size, hidden_size)
        self.fc_hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_output = torch.nn.Linear(hidden_size, 1)

        self.n_actions = n_actions

    def forward(self, actions, x) -> torch.tensor:
        x = torch.cat((x, actions), dim=-1)

        x = self.fc_input(x)
        x = self.act(x)
        x = self.fc_hidden(x)
        x = self.act(x)
        x = self.fc_output(x)

        return x

    def predict(self, x) -> torch.tensor:
        device = next(iter(self.parameters())).device

        actions_weight = torch.zeros(x.shape[0], self.n_actions).to(device)

        for i in range(self.n_actions):
            actions = torch.zeros(x.shape[0], self.n_actions).to(device)
            actions[:, i] = 1.

            predicted = self.forward(actions, x)

            actions_weight[:, i] = predicted

        return self.softmax(actions_weight)


class Sample:
    def __init__(self, x: list, y: float, action_id):
        self.X = x  # ladars values
        self.y = y  # rewards
        self.action_id = action_id  # action

    def get_processed_sample(self, n_actions) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Get processed sample
        :param n_actions: num of actions
        :return: 3d-tuple with tensors
        """
        action = torch.zeros(n_actions)
        action[self.action_id] = 1

        action = action.unsqueeze(0).unsqueeze(0)

        x = torch.as_tensor(self.X, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        y = torch.as_tensor([self.y], dtype=torch.float).unsqueeze(0).unsqueeze(0)

        return x, y, action


class AIAgent:
    def __init__(self,
                 n_actions: int,
                 input_size: int,
                 hidden_size: int,
                 gpu_mode: bool = False,
                 model_path: str = None,
                 load: bool = False,
                 train: bool = True):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_mode else 'cpu')
        self.model_path = model_path

        self.nn = NN(n_actions, input_size, hidden_size)

        if load:
            self.nn.load_state_dict(torch.load(model_path))

        self.nn.to(self.device)

        if train:
            self.nn.train()
        else:
            self.nn.eval()

        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.0001)
        self.loss = torch.nn.MSELoss()

    def save(self):
        """
        Save nn model
        :return: None
        """
        torch.save(self.nn.state_dict(), self.model_path)

    def step(self, sample: Sample):
        x, y, actions = sample.get_processed_sample(self.nn.n_actions)

        x = x.to(self.device)
        y = y.to(self.device)
        actions = actions.to(self.device)

        self.optimizer.zero_grad()

        predicted = self.nn.forward(actions, x)

        loss_value = self.loss(predicted, y)

        loss_value.backward()

        self.optimizer.step()

        self.save()

    def predict(self, x: list) -> int:
        """
        Use NN to predict action
        :param x: ladar values (list)
        :return: action id (int)
        """
        x = torch.as_tensor(x).to(self.device)
        x = x.unsqueeze(0)

        result = self.nn.predict(x).squeeze(0).cpu().detach().numpy()

        return choice(self.nn.n_actions, p=result)
