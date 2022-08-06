import torch

from numpy.random import choice

__all__ = ['AIAgent', 'Data']


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

            actions_weight[:, i] = self.forward(actions, x)

        return self.softmax(actions_weight)


class Data:
    def __init__(self):
        self.action_ids = []  # actions
        self.X = []  # ladars values
        self.y = []  # rewards

    def add_row(self, action_id: int, X: list, y: float):
        """
        Add row to data
        :param action_id: action (integer)
        :param X: ladar values (list)
        :param y: rewards float
        :return: None
        """
        self.action_ids.append(action_id)
        self.X.append(X)
        self.y.append(y)

    def get_processed_data(self, n_actions) -> (torch.tensor, torch.tensor, torch.tensor):
        actions = torch.zeros(len(self.action_ids), n_actions)
        for i, row in enumerate(self.action_ids):
            actions[i, row] = 1.

        actions = actions.unsqueeze(0)

        X = torch.as_tensor(self.X).unsqueeze(0)
        y = torch.as_tensor(self.y).unsqueeze(0)

        # shuffle data
        shuffle = torch.randperm(X.size()[1])

        X = X[:, shuffle]
        y = y[:, shuffle]
        actions = actions[:, shuffle]

        return X, y, actions

    def clear_all_data(self):
        self.X = []
        self.y = []
        self.action_ids = []


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

        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.01)
        self.loss = torch.nn.MSELoss()

    def save(self):
        """
        Save nn model
        :return: None
        """
        torch.save(self.nn.state_dict(), self.model_path)

    def step(self, data: Data):
        X, y, actions = data.get_processed_data(self.nn.n_actions)

        X = X.to(self.device)
        y = y.to(self.device)
        actions = actions.to(self.device)

        last_loss = None

        for _ in range(3):
            self.optimizer.zero_grad()

            predicted = self.nn.forward(actions, X)

            loss_value = self.loss(predicted.squeeze(2), y)

            loss_value.backward()

            self.optimizer.step()

            last_loss = loss_value.item()

        print(f'loss {last_loss}')

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
