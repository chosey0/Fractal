import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size*2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=input_size*2, out_channels=input_size*4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=input_size*4, out_channels=input_size*8, kernel_size=1, padding=0)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(input_size*8*8, input_size*8*4) # Adjust this calculation based on the pooling layers
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size*8*4, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv1_x = self.pool(torch.relu(self.conv1(x)))
        conv2_x = self.pool(torch.relu(self.conv2(conv1_x)))
        conv3_x = self.pool(torch.relu(self.conv3(conv2_x)))
        fc1_x = self.fc1(conv3_x.flatten(1))
        fc2_x = self.fc2(self.relu(fc1_x))

        return fc2_x
        
    def register_hooks(self):
        self.conv1.register_forward_hook(self.get_activation('conv1'))
        self.conv2.register_forward_hook(self.get_activation('conv2'))
        self.conv3.register_forward_hook(self.get_activation('conv3'))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
            
        return hook