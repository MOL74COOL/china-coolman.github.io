from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):

    def __init__(self,base_model, input_size, num_classes, num_channels,
                 kernel_size=2, dropout=0.3, tied_weights=False):
        super(TCN, self).__init__()
        self.base_model = base_model
        for param in base_model.parameters():
            param.requires_grad = (True)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        raw_outputs =self.base_model(**input)
        tokens = raw_outputs.last_hidden_state
        y = self.tcn(tokens.transpose(1, 2)).transpose(1, 2)

        # Self-Attention
        K = self.key_layer(y)
        Q = self.query_layer(y)
        V = self.value_layer(y)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)
        attention_output = torch.mean(attention_output, dim=1)
        predicts = self.block(attention_output)
        return predicts
