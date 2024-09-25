# Define the Config class to hold configuration parameters
class Config:
    def __init__(self, seed, ann_in_dim, hidden_layers, loss_str, data_source, num_epochs, log_interval, alpha=None, beta=None, patience=10, min_delta=0.001,
                 learning_rate=1e-4, weight_decay=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, decay_rate=0.9, decay_steps=2000, batch_size=64):
        """
        Configuration for the model and training process.

        Parameters:
        - seed: Random seed for initialization.
        - ann_in_dim: Input dimension for the ANN model.
        - hidden_layers: List of integers representing the number of units in each hidden layer.
        - loss_str: String identifier for the loss function.
        - data_source: Identifier for the data source.
        - num_epochs: Number of epochs for training.
        - log_interval: Interval for logging training progress.
        - alpha: Weight for data fitting loss.
        - beta: Weight for PDE loss.
        - patience: Number of epochs to wait for improvement before stopping.
        - min_delta: Minimum change in validation loss to qualify as an improvement.
        - learning_rate: Learning rate for the optimizer.
        - weight_decay: L2 regularization term.
        - beta1: First moment decay rate for the Adam optimizer.
        - beta2: Second moment decay rate for the Adam optimizer.
        - eps: Epsilon value to prevent division by zero in the optimizer.
        - decay_rate: Rate at which the learning rate decays.
        - decay_steps: Number of steps before applying decay to the learning rate.
        """
        self.seed = seed
        self.ann_in_dim = ann_in_dim
        self.hidden_layers = hidden_layers
        self.loss_str = loss_str
        self.data_source = data_source
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.alpha = alpha
        self.beta = beta
        self.patience = patience
        self.min_delta = min_delta
        
        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        # batch size
        self.batch_size = batch_size