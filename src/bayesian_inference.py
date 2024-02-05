from network import *
from utils import *
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

class BayesianMLP(nn.Module):
    """
    Implement a logistic regression classifiers
    """

    def __init__(self, options={'d': 20, 'o':10}):
        super(BayesianLR, self).__init__()
        self.input_layer = nn.Linear(options['d'], 10)
        self.hidden_layer = nn.Linear(10, options['o'])
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.input_layer(x)
        output = self.hidden_layer(output)
        return self.tanh(output)

class BayesianLR(nn.Module):
    """
    Implement a logistic regression classifiers
    """

    def __init__(self, options={'d': 20, 'o':10}):
        super(BayesianLR, self).__init__()
        self.input_layer = nn.Linear(options['d'], options['o'])
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.input_layer(x)
        return self.tanh(output) # because we know that the output is in the range [-1,1]

def train_bayesian_lr(train_loader, options = {'epochs':300, 'lr':1e-3, 'R': [],'U': []}):
    """
    Train a Bayesian regressor model,
    :params: train_loader, data loader that contrain input and multi-output
    :params: options, dictionary provide the settings of training such as learning rate
    return: a model
    """
    R, U = options['R'], options['U']
    model = BayesianLR(options={'d': len(R), 'o': len(U)})
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), options['lr'])

    for _ in range(options['epochs']):
        for x, y in train_loader:
            output = model(x)
            model.zero_grad()
            kl = get_kl_loss(model)
            mse_loss = criterion(output, y)
            loss = mse_loss + kl / len(y)

            loss.backward()
            optimizer.step()

    return model

def infer_bayesian_lr(model, x,R, n_mc = 100):
    """
    Perform MCMC inference
    R is the sorted list of indicies revealed features,
    """
    model.eval()
    d = max(x.shape)-len(R)
    y_pred_list = []
    x = x.reshape(-1)
    with torch.no_grad():
            for _ in range(n_mc):
                y_pred = model(torch.Tensor(x[np.array(R)])).detach().numpy()
                y_pred_list.append(y_pred)

    return np.array(y_pred_list).reshape(n_mc, d)

def construct_bayes_dict(x_train, public, private):
    """
    Construct a dictionary of bayesian regression models to predict X_U given X_R =x_R

    :param x_train: nxd numpy matrix, where  n is the number of training samples, d is the number of features
    :param public: list of indices public features e..g, [0, 4, 6]
    :param private:  list of indices of private features [1,2,3, 5]
    :return:
    """
    S_all = powerset(private)
    bayes_dict = {}
    d = x_train.shape[1]
    for S in S_all:
        if len(S) != len(private):
            R = tuple(sorted(public + S))
            U = [x for x in range(d) if x not in R] # this is already sorted
            train_tensor = TensorDataset(torch.Tensor(x_train[:, R]), torch.Tensor(x_train[:, U]))
            train_loader = DataLoader(dataset=train_tensor, batch_size=32, shuffle=True)
            model = train_bayesian_lr(train_loader, options={'epochs': 200, 'lr': 1e-2, 'R': R, 'U': U})
            bayes_dict[R] = {'U': U, 'model': model}

    return bayes_dict