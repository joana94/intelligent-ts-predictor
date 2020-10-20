import torch 
from torch import nn, optim


class TradRNN(nn.Module):
    """
    Class to initialize and instantiate an Elman's (traditional) Recurrent Neural Network.
        
    Arguments of class constructor:
    -------------------------------
    input_size: int
        The number of features of the time series. In a univariate time series setting is usually 1.
    hidden_size: int
        The number of hidden neurons in the recurrent unit. The output shape of the recurrent unit changes based on this value.
    n_layers: int
        The number of stacked recurrent units.
    output_size: int
        Related to the number of features to be predicted. For univariate time series is usually 1.
    seq_len: int
        The sequence length of each of the input sequences.
    use_all_h: bool, default: True
        Whether to use the information from all the hidden states. If False, only the last hidden state is used.
    dropout_p: float, greater than zero and less than 1, default: 0.1
        Acts as a regularization technique that helps avoiding overfitting. 
        It may also be used as an indicator of model uncertainty.
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size, seq_len, use_all_h=True, dropout_p = 0.1, device='auto'):
   
        super(TradRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len

        # Dropout is only alllowed for stacked rnns (two or more layers)
        if self.n_layers ==1:
              self.dropout = 0
        else:
            self.dropout = dropout_p
        self.use_all_h = use_all_h

        # Automatically choose device used for tensor calculations
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == 'cpu' or device =='gpu':
            self.device = device
        else:
            self.device = None
        # the input_size is the number of features per timestep
        # hidden_size is the number of nodes in each timestep

        # the inputs passed into the RNN must have shape [batch_size, seq_len, input_size]
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, dropout=self.dropout)

        # the outputs of the RNN must be passed to a fully connected layer that transforms 
        # them to a more desirable output shape
        if self.use_all_h:
            # uses information from every hidden state
            # passes the information through a fully connected layer
            self.fc = nn.Linear(hidden_size*self.seq_len, output_size) 
        else:
            # uses only information from the last hidden state
            # passes the information through a fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 -> hidden state shape [n_layers, batch_size, hidden_size]
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward pass
        output, _ = self.rnn(x, h0)

        if self.use_all_h:
            output = output.reshape(output.shape[0], -1) # out.shape[0] -> batch_size, -1-> hidden_size*seq_len
            output = self.fc(output)
        else:
            output = self.fc(output[:, -1, :])

        return output

class GRU(nn.Module):

    """
    Class to initialize and instantiate a Gated Recurrent Unit (GRU) Neural Network.
        
    Arguments of class constructor:
    -------------------------------
    input_size: int
        The number of features of the time series. In a univariate time series setting is usually 1.
    hidden_size: int
        The number of hidden neurons in the recurrent unit. The output shape of the recurrent unit changes based on this value.
    n_layers: int
        The number of stacked recurrent units.
    output_size: int
        Related to the number of features to be predicted. For univariate time series is usually 1.
    seq_len: int
        The sequence length of each of the input sequences.
    use_all_h: bool, default: True
        Whether to use the information from all the hidden states. If False, only the last hidden state is used.
    dropout_p: float, greater than zero and less than 1, default: 0.1
        Acts as a regularization technique that helps avoiding overfitting. 
        It may also be used as an indicator of model uncertainty.
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size, seq_len, use_all_h=True, dropout_p = 0.1, device='auto'):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        # Dropout is only alllowed for stacked rnns (two or more layers)
        if self.n_layers ==1:
              self.dropout = 0
        else:
            self.dropout = dropout_p
        self.use_all_h = use_all_h

        # Automatically choose device used for tensor calculations
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == 'cpu' or device =='gpu':
            self.device = device
        else:
            self.device = None
        # the input_size is the number of features per timestep
        # hidden_size is the number of nodes in each timestep

        # the inputs passed into the RNN must have shape [batch_size, seq_len, input_size]
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=self.dropout)

        # the outputs of the RNN must be passed to a fully connected layer that transforms 
        # them to a more desirable output shape
        if self.use_all_h:
            # uses information from every hidden state
            # passes the information through a fully connected layer
            self.fc = nn.Linear(hidden_size*self.seq_len, output_size) 
        else:
            # uses only information from the last hidden state
            # passes the information through a fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 -> hidden state shape [n_layers, batch_size, hidden_size]
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward pass
        output, _ = self.gru(x, h0)
        if self.use_all_h:
            output = output.reshape(output.shape[0], -1) # out.shape[0] -> batch_size, -1-> hidden_size*seq_len
            output = self.fc(output)
        else:
            output = self.fc(output[:, -1, :])

        return output

class LSTM(nn.Module):
    """
    Class to initialize and instantiate a Long Short-Term Memory (LSTM) Neural Network.
        
    Arguments of class constructor:
    -------------------------------
    input_size: int
        The number of features of the time series. In a univariate time series setting is usually 1.
    hidden_size: int
        The number of hidden neurons in the recurrent unit. The output shape of the recurrent unit changes based on this value.
    n_layers: int
        The number of stacked recurrent units.
    output_size: int
        Related to the number of features to be predicted. For univariate time series is usually 1.
    seq_len: int
        The sequence length of each of the input sequences.
    use_all_h: bool, default: True
        Whether to use the information from all the hidden states. If False, only the last hidden state is used.
    dropout_p: float, greater than zero and less than 1, default: 0.1
        Acts as a regularization technique that helps avoiding overfitting. 
        It may also be used as an indicator of model uncertainty.
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size, seq_len, use_all_h=True, dropout_p = 0.1, device='auto'):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len

        # Dropout is only alllowed for stacked rnns (two or more layers)
        if self.n_layers ==1: 
              self.dropout = 0
        else:
            self.dropout = dropout_p
        self.use_all_h = use_all_h
        # Automatically choose device used for tensor calculations
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == 'cpu' or device =='gpu':
            self.device = device
        else:
            self.device = None
        # the input_size is the number of features per timestep
        # hidden_size is the number of nodes in each timestep

        # the inputs passed into the RNN must have shape [batch_size, seq_len, input_size]
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=self.dropout)

        # the outputs of the RNN must be passed to a fully connected layer that transforms 
        # them to a more desirable output shape
        if self.use_all_h:
            # uses information from every hidden state
            # passes the information through a fully connected layer
            self.fc = nn.Linear(hidden_size*self.seq_len, output_size) 
        else:
            # uses information only from the last hidden state
            # passes the information through a fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 -> hidden state shape [n_layers, batch_size, hidden_size]
        # initialized with zeros for every sample or batch of samples
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        
        # c0 -> cell state shape [n_layers, batch_size, hidden_size]
        # initialized with zeros for every sample or batch of samples
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward pass
        output, _ = self.lstm(x, (h0, c0))
        if self.use_all_h:
            output = output.reshape(output.shape[0], -1) # out.shape[0] -> batch_size, -1-> hidden_size*seq_len
            output = self.fc(output)
        else:
            output = self.fc(output[:, -1, :]) # only the last hidden state

        return output
        