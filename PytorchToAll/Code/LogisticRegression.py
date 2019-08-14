import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.],[0.],[1.], [1.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1000):
    y_pred = model(x_data) #call forward
    loss = criterion(y_pred, y_data) #compute the loss given model output prediction
    print(epoch, loss.data.item()) #print epoch and loss value

    optimizer.zero_grad() #clear gradient before loss backward
    loss.backward() # calculates gradients
    optimizer.step() #updates weights given gradients
  
pred_var = Variable(torch.Tensor([[1.0]]))
print("Predict 1", 1.0, model.forward(pred_var).data.item() > 0.5)

pred_var = Variable(torch.Tensor([[7.0]]))
print("Predict 1", 7.0, model.forward(pred_var).data.item() > 0.5)