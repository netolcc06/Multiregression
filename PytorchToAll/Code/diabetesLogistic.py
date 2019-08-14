import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

xy = np.loadtxt('./Data/diabetes.csv', delimiter = ',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:,0:-1]))
y_data = Variable(torch.from_numpy(xy[:,[-1]]))

print(x_data.shape)
print(y_data.shape)

class diabetesLogistic(torch.nn.Module):
    def __init__(self):
        super(diabetesLogistic, self).__init__()
        self.l1 = torch.nn.Linear(8,6)
        self.l2 = torch.nn.Linear(6,4)
        self.l3 = torch.nn.Linear(4,1)

        self.sigmoid = F.sigmoid

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = diabetesLogistic()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(100):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()