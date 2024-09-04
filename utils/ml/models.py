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
    
class Net3D_Positive(torch.nn.Module):
    
    def __init__(self, out_dim):
        super(Net3D_Positive, self).__init__()
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
            ),
            torch.nn.ReLU()      
        )
        self.net.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x):
        return self.net(x)

class Net3D_wDO(torch.nn.Module):
    
    def __init__(self, out_dim):
        super(Net3D_wDO, self).__init__()
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
            torch.nn.Dropout(
                p = 0.2
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features = 24,
                out_features = 12
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(
                p = 0.2    
            ),
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
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            
    def forward(self, x):
        return self.net(x)


class Net3D_v1(torch.nn.Module):
    
    def __init__(self, out_dim):
        super(Net3D_v1, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels = 1,
                out_channels = 6,
                kernel_size = [5,5,5]
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),            
            torch.nn.Conv3d(
                in_channels = 6,
                out_channels = 12,
                kernel_size = [3,3,3]
            ),
            torch.nn.ReLU(),            
            torch.nn.Conv3d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = [3,3,3]
            ),
            torch.nn.ReLU(),            
            torch.nn.Conv3d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = [3,3,3]
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),
            torch.nn.Conv3d(
                in_channels = 12,
                out_channels = 24,
                kernel_size = [3,3,3]
            ),
            torch.nn.ReLU(),
            torch.nn.Conv3d(
                in_channels = 24,
                out_channels = 24,
                kernel_size = [3,3,3]
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features = 24,
                out_features = 12
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features = 12,
                out_features = 6
            ),
            torch.nn.ReLU(),
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

class Net3D_v2(torch.nn.Module):
    
    def __init__(self, out_dim):
        super(Net3D_v2, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels = 1,
                out_channels = 8,
                kernel_size = [6,5,5]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),            
            torch.nn.Conv3d(
                in_channels = 8,
                out_channels = 16,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),            
            torch.nn.Conv3d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),            
            torch.nn.Conv3d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),
            torch.nn.Conv3d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = [3,3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.Conv3d(
                in_channels = 128,
                out_channels = 128,
                kernel_size = [1,3,3]
            ),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(
                kernel_size = [2,2,2],
                stride = 2
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features = 128,
                out_features = 64
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features = 64,
                out_features = 32
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features = 32,
                out_features = out_dim
            )      
        )
        self.net.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x):
        return self.net(x)

class Net3D_v3(torch.nn.Module):
    
    def __init__(self, out_dim):
        super(Net3D_v3, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=[6, 5, 5]
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size=[2, 2, 2],
                stride=2
            ),            
            torch.nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=[3, 3, 3]
            ),
            torch.nn.ReLU(),            
            torch.nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=[3, 3, 3]
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size=[2, 2, 2],
                stride=2
            ),
            torch.nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=[3, 3, 3]
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(
                kernel_size=[2, 2, 2],
                stride=2
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=64 * 1 * 2 * 2 *2,
                out_features=128
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=128,
                out_features=64
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=64,
                out_features=out_dim
            )      
        )
    
    def forward(self, x):
        return self.net(x)

