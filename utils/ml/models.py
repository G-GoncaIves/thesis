import torch

class Net2D(torch.nn.Module):
    
    def __init__(self, out_dim=1):
        super(Net2D, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels = 1,
                out_channels = 6,
                kernel_size = [5,5]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(
                kernel_size = [2,2],
                stride = 2
            ),            
            torch.nn.Conv2d(
                in_channels = 6,
                out_channels = 12,
                kernel_size = [3,3]
            ),
            torch.nn.Tanh(),            
            torch.nn.Conv2d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = [3,3]
            ),
            torch.nn.Tanh(),            
            torch.nn.Conv2d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = [3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(
                kernel_size = [2,2],
                stride = 2
            ),
            torch.nn.Conv2d(
                in_channels = 12,
                out_channels = 24,
                kernel_size = [3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.Conv2d(
                in_channels = 24,
                out_channels = 24,
                kernel_size = [3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(
                kernel_size = [2,2],
                stride = 2
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features = 24,
                out_features = 12
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features = 12,
                out_features = 6
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features = 6,
                out_features = out_dim
            )      
        )
        self.net.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x):
        return self.net(x)
    
    
class Net3D(torch.nn.Module):
    
    def __init__(self, out_dim):
        super(Net3D, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels = 1,
                out_channels = 6,
                kernel_size = [5,5,5]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),            
            torch.nn.Conv3d(
                in_channels = 6,
                out_channels = 12,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),            
            torch.nn.Conv3d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),            
            torch.nn.Conv3d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),
            torch.nn.Conv3d(
                in_channels = 12,
                out_channels = 24,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.Conv3d(
                in_channels = 24,
                out_channels = 24,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features = 24,
                out_features = 12
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features = 12,
                out_features = 6
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features = 6,
                out_features = out_dim
            )      
        )
        self.net.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x):
        return self.net(x)
