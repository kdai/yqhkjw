# implement two networks:
# 1. f(x): ranking function
# 2. h(xi, xj): checking function
import torch
import torch.nn as nn
import torch.nn.functional as FF
import pdb
from opts import args
from torch.nn import init
def addGaussianNoise(x,sigma=0.05):
    if random.random() < 0.5:
        return x
    return x+torch.zeros_like(x.data).normal_()*sigma

#FNet+HNet = mAP 0.51
class FNet(nn.Module):
    def __init__(self, feature_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        out = self.relu(self.dropout1(self.fc1(x)))
        out = self.relu(self.dropout2(self.fc2(out)))
        out = self.fc3(out)
        # out = self.softmax(out)[:,1].contiguous()
        return out  # [n, 8,2]  

class AttentionUnit(nn.Module):
    def __init__(self,feature_dim = 128):
        super().__init__()
        self.fc1 = nn.Linear(2*feature_dim,feature_dim)
        self.fc2 = nn.Linear(feature_dim,feature_dim)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.relu(self.dropout(self.fc1(x)))
        output = self.relu(self.fc2(output))
        return output
class AttentionModule(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        self.Conv = torch.nn.Sequential()
        self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit(feature_dim))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax).unsqueeze(1)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax).unsqueeze(1)),1)
        #attention fusion
        attention = F.softmax(concat,1)
        b,c,w = attention.shape
        vx = vx.unsqueeze(1).expand(b,c,w)
        vx_att = vx*attention
        vx_att = vx_att.permute(0,2,1)
        b,w,c = vx_att.shape
        vx_att = vx_att.unsqueeze(3).expand(b,w,c,c)
        vx_att = self.relu(self.Conv(vx_att).view(b,w))
        return vx_att
#FNet+HNet = mAP 0.51
class FNet1(nn.Module):
    def __init__(self, feature_dim = 512):
        super().__init__()
        self.att_net = AttentionModule(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        out1 = self.relu(self.dropout1(self.fc1(x)))

        # att = self.att_net(out1)
        # att_x = torch.mul(att,out1)
        att_out = self.relu(self.dropout2(self.fc2(out1)))
        att_out = self.fc3(att_out)

        out = self.relu(self.dropout2(self.fc2(out1)))
        out = self.fc3(out)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,out  # [n, 8,2]  


class HNet(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(input_dim, 512)
        
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x [n, 8, 1024]
        # x = torch.cat((xi, xj)).view(-1, 1024)    
        out = self.relu(self.dropout1(self.fc1(x)))
        out = self.relu(self.dropout2(self.fc2(out)))
        out = self.fc3(out).view(-1,args.num_per_group).contiguous()
        out = self.softmax(out)

        return out

class HNet1(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(input_dim, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # init.constant(self.bn1.weight, 1)
        # init.constant(self.bn1.bias, 0)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        init.constant(self.bn2.weight, 1)
        init.constant(self.bn2.bias, 0)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x [n, 8, 1024]
        # x = torch.cat((xi, xj)).view(-1, 1024)    
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out).view(-1,args.num_per_group).contiguous()
        out = self.softmax(out)

        return out

class HNet2(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(input_dim, 512)
        
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x [n, 8, 1024]
        # x = torch.cat((xi, xj)).view(-1, 1024)    
        out = self.relu(self.dropout1(self.fc1(x)))
        out = self.relu(self.dropout2(self.fc2(out)))
        out = self.fc3(out).view(-1,args.num_per_group).contiguous()
        out = self.softmax(out)

        return out

