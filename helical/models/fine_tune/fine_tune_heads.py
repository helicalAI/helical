import torch

class ClassificationHead(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassificationHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = torch.nn.Dropout(p=0.02)
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, inputs):
        drop = self.dropout(inputs)
        output = self.linear(drop)
        return output