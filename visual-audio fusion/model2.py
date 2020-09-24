# implement two networks:
# 1. f(x): ranking function
# 2. h(xi, xj): checking function
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.relu(self.dropout(self.fc1(x)))
        output = self.relu(self.dropout(self.fc2(output)))
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
        # attention = F.softmax(torch.log(concat+1e-9),1)
        attention = concat
        # attention = F.softmax(concat)
        # pdb.set_trace()
        b,c,w = attention.shape
        vx = vx.unsqueeze(1).expand(b,c,w)
        vx_att = vx*attention
        vx_att = vx_att.permute(0,2,1)
        b,w,c = vx_att.shape
        vx_att = vx_att.unsqueeze(3).expand(b,w,c,c)
        vx_att = self.relu(self.Conv(vx_att).view(b,w))

        return vx_att

class AttentionUnit2(nn.Module):
    def __init__(self,feature_dim = 128):
        super().__init__()
        self.fc1 = nn.Linear(2*feature_dim,feature_dim)
        self.fc2 = nn.Linear(feature_dim,feature_dim)
        # self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        return output
class AttentionModule2(nn.Module):
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
            self.attentionUnits.append(AttentionUnit2(feature_dim))
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
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = concat
        attention = F.softmax(concat)
        # pdb.set_trace()
        b,c,w = attention.shape
        vx = vx.unsqueeze(1).expand(b,c,w)
        vx_att = vx*attention
        vx_att = vx_att.permute(0,2,1)
        b,w,c = vx_att.shape
        vx_att = vx_att.unsqueeze(3).expand(b,w,c,c)
        vx_att = self.relu(self.Conv(vx_att).view(b,w))
        return vx_att


#multi head,dropout
class AttentionUnit3(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.dropout(self.relu(self.fc1(x)))
        output =self.dropout(self.relu(self.fc2(output)))
        return output
class AttentionModule3(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit3(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        # vx_att = vx+concat
        return concat

#multi head,bn
class AttentionUnit3_1(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.bn1 = nn.BatchNorm1d(in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output =self.bn1(self.relu(self.fc1(x)))
        output =self.relu(self.fc2(output))
        return output
class AttentionModule3_1(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit3_1(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        # vx_att = vx+concat
        return concat

#multi head ,没有dropout\bn
class AttentionUnit3_2(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.bn1 = nn.BatchNorm1d(in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output =self.relu(self.fc1(x))
        output =self.relu(self.fc2(output))
        return output
class AttentionModule3_2(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit3_2(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        # vx_att = vx+concat
        return concat


class AttentionUnit4(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.relu(self.dropout(self.fc1(x)))
        output =self.dropout(self.fc2(output))
        return output
class AttentionModule4(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit4(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        vx_att = vx+concat
        ax_att = ax+concat
        return vx_att,ax_att

#multi head
class AttentionUnit5(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)
        # self.dropout = torch.nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(in_feature_dim)
        # init.constant(self.bn3.weight, 1)
        # init.constant(self.bn3.bias, 0)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.relu(self.fc1(x))
        output = self.bn(output)
        output =self.relu(self.fc2(output))
        return output
class AttentionModule5(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit5(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        vx_att = vx+concat
        return vx_att

#multi head
class AttentionUnit6(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.afc = nn.Linear(in_feature_dim,64)
        self.vfc = nn.Linear(in_feature_dim,64)
        self.fusion = nn.Linear(64,out_feature_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        # x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        vx = self.bn1(self.relu(self.vfc(vx)))
        ax = self.bn2(self.relu(self.afc(ax)))
        fx = self.relu(self.fusion(vx+ax))
        return fx
class AttentionModule6(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit6(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        # vx_att = vx+concat
        return concat




#FNet+HNet = mAP 0.51
class FNet1(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        # self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        
        att_x = self.att_net(vo1,ax)

        att_out = self.relu(self.dropout2(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

class FNet2(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        init.constant(self.bn1.weight, 1)
        init.constant(self.bn1.bias, 0)

        self.bn2 = nn.BatchNorm1d(64)
        init.constant(self.bn2.weight, 1)
        init.constant(self.bn2.bias, 0)

        self.bn3 = nn.BatchNorm1d(64)
        init.constant(self.bn3.weight, 1)
        init.constant(self.bn3.bias, 0)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.fc1(vx))

        att_x = self.att_net(vo1,ax)

        att_out = self.relu(self.bn1(self.fc2(att_x)))
        att_out = self.fc3(att_out)

        vout = self.relu(self.bn2(self.fc2(vo1)))
        vout = self.fc3(vout)

        aout = self.relu(self.bn3(self.fc2(ax)))
        aout = self.fc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

class FNet3(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.v_att_net = AM(128)
        self.a_att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fusion = nn.Linear(128*2, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        init.constant(self.bn1.weight, 1)
        init.constant(self.bn1.bias, 0)

        self.bn2 = nn.BatchNorm1d(64)
        init.constant(self.bn2.weight, 1)
        init.constant(self.bn2.bias, 0)

        self.bn3 = nn.BatchNorm1d(64)
        init.constant(self.bn3.weight, 1)
        init.constant(self.bn3.bias, 0)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.fc1(vx))

        v_att_x = self.v_att_net(vo1,ax)
        a_att_x = self.a_att_net(ax,vo1)
        att_x = torch.cat((v_att_x,a_att_x),1)
        att_x = self.relu(self.fusion(att_x))
        att_out = self.relu(self.bn1(self.fc2(att_x)))
        att_out = self.fc3(att_out)

        vout = self.relu(self.bn2(self.fc2(vo1)))
        vout = self.fc3(vout)

        aout = self.relu(self.bn3(self.fc2(ax)))
        aout = self.fc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

class FNet4(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.v_att_net = AM(128)
        self.a_att_net = AM(128)
        self.fusion = nn.Linear(128*2, 128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)

        self.afc2 = nn.Linear(128, 64)
        self.afc3 = nn.Linear(64, 1)

        self.vfc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))

        v_att_x = self.v_att_net(vo1,ax)
        a_att_x = self.a_att_net(ax,vo1)
        att_x = torch.cat((v_att_x,a_att_x),1)
        att_x = self.relu(self.fusion(att_x))
        att_out = self.relu(self.dropout2(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

class FNet5(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        # self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
       
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.fc1(vx))

        att_x = self.att_net(vo1,ax)

        att_out = self.relu(self.fc2(att_x))
        att_out = self.fc3(att_out)

        vout = self.relu(self.fc2(vo1))
        vout = self.fc3(vout)

        aout = self.relu(self.fc2(ax))
        aout = self.fc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  
#一个AM，audio-guide attention
class FNet6(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))

        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention, 统回归器
class FNet6_1(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.afc(ax))

        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.relu(self.dropout2(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.attfc2(vo1)))
        vout = self.attfc3(vout)

        aout = self.relu(self.dropout2(self.attfc2(ax)))
        aout = self.attfc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention, bn
class FNet6_2(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.afc = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.attfc2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)

        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.bn1(self.relu(self.fc1(vx)))
        ax = self.bn2(self.relu(self.afc(ax)))

        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.bn3(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.bn4(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.bn5(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class FNet6_3(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.fc1(vx))
        ax = self.relu(self.afc(ax))

        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.relu(self.attfc2(att_x))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.vfc2(vo1))
        vout = self.vfc3(vout)

        aout = self.relu(self.afc2(ax))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class FNet6_4(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))

        att_x = self.att_net(vo1,ax)
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class FNet6_5(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.attfc2 = nn.Linear(128, 64)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)

        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.bn1(self.relu(self.fc1(vx)))
        ax = self.bn2(self.relu(self.afc(ax)))
        
        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.dropout1(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.attfc2(vo1)))
        vout = self.attfc3(vout)

        aout = self.dropout3(self.relu(self.attfc2(ax)))
        aout = self.attfc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  


class FNet7(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        att_x = self.att_net(vo1,ax)
        att_x = self.att_net(vo1,ax)


        att_out = self.relu(self.dropout2(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，将att_audio和att_visul分别做分数预测
class FNet8(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.visual_attfc2 = nn.Linear(128, 64)
        self.visual_attfc3 = nn.Linear(64, 1)
        self.audio_attfc2 = nn.Linear(128, 64)
        self.audio_attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.fc1(vx))
        ax = self.relu(self.afc(ax))

        visual_att_x,audio_att_x = self.att_net(vo1,ax)
        
        visual_att_out = self.relu(self.dropout2(self.visual_attfc2(visual_att_x)))
        visual_att_out = self.visual_attfc3(visual_att_out)

        audio_att_out = self.relu(self.dropout2(self.audio_attfc2(audio_att_x)))
        audio_att_out = self.audio_attfc3(audio_att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return visual_att_out,audio_att_out,vout,aout,visual_att_x,audio_att_x,vo1,ax  # [n, 8,2]  

#一个AM，将att_audio和att_visul做fusion
class FNet9(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.fusion = nn.Linear(2*128,128)
        self.attfc2 = nn.Linear(128, 64)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.afc(ax))

        visual_att_x,audio_att_x = self.att_net(vo1,ax)
        attx = torch.cat((visual_att_x,audio_att_x),1)
        attx = self.relu(self.fusion(attx))
     

        att_out = self.relu(self.dropout2(self.attfc2(attx)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,attx,vo1,ax  # [n, 8,2]  

#一个AM，没做fusion，直接将attention之后的audio_att当成audio
class FNet10(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.dropout1(self.afc(ax)))

        att_x,ax = self.att_net(vo1,ax)

        att_out = self.relu(self.dropout2(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#两个AM，将att_audio和att_visul做fusion
class FNet11(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.fusion = nn.Linear(2*128,128)
        self.attfc2 = nn.Linear(128, 64)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.dropout1(self.afc(ax)))

        visual_att_x = self.visual_att_net(vo1,ax)
        audio_att_x = self.audio_att_net(ax,vo1)

        attx = torch.cat((visual_att_x,audio_att_x),1)
        attx = self.relu(self.fusion(attx))
     

        att_out = self.relu(self.dropout2(self.attfc2(attx)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,attx,vo1,ax  # [n, 8,2]  

#一个AM，visual-guide attention
class FNet12(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.afc(ax))

        att_x = self.att_net(ax,vo1)

        att_out = self.relu(self.dropout2(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#两个AM，将att_audio和att_visul做fusion
class FNet13(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.WA = nn.Linear(128,128)
        self.WV = nn.Linear(128,128)
        self.fusion = nn.Linear(128,128)

        self.attfc2 = nn.Linear(128, 64)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.dropout1(self.afc(ax)))

        visual_att_x = self.visual_att_net(vo1,ax)
        audio_att_x = self.audio_att_net(ax,vo1)

        # attx = torch.cat((visual_att_x,audio_att_x),1)
        attx = self.relu(self.fusion(self.WA(audio_att_x)+self.WV(visual_att_x)))
        # attx = self.relu(self.fusion(attx))

        att_out = self.relu(self.dropout2(self.attfc2(attx)))
        att_out = self.attfc3(att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,attx,vo1,ax  # [n, 8,2]  

#两个AM，将att_audio和att_visul做fusion,做bn
class FNet13_1(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.WA = nn.Linear(128,128)
        self.WV = nn.Linear(128,128)
        self.fusion = nn.Linear(128,128)

        self.attfc2 = nn.Linear(128, 64)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.fc1(vx))
        ax = self.relu(self.afc(ax))

        visual_att = self.visual_att_net(vo1,ax)
        audio_att = self.audio_att_net(ax,vo1)
        visual_att_x = vo1+visual_att
        audio_att_x = ax+audio_att

        # attx = torch.cat((visual_att_x,audio_att_x),1)
        attx = self.relu(self.fusion(self.WA(audio_att_x)+self.WV(visual_att_x)))
        # attx = self.relu(self.fusion(attx))

        att_out = self.bn1(self.relu(self.attfc2(attx)))
        att_out = self.attfc3(att_out)

        vout = self.bn2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.bn3(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,attx,vo1,ax  # [n, 8,2]  

#两个AM，将att_audio和att_visul做fusion,共用一个回归器
class FNet13_2(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.WA = nn.Linear(128,128)
        self.WV = nn.Linear(128,128)
        self.fusion = nn.Linear(128,128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
      

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.dropout1(self.afc(ax)))

        visual_att_x = self.visual_att_net(vo1,ax)
        audio_att_x = self.audio_att_net(ax,vo1)

        # attx = torch.cat((visual_att_x,audio_att_x),1)
        attx = self.relu(self.fusion(self.WA(audio_att_x)+self.WV(visual_att_x)))
        # attx = self.relu(self.fusion(attx))

        att_out = self.relu(self.dropout2(self.fc2(attx)))
        att_out = self.fc3(att_out)

        vout = self.relu(self.dropout2(self.fc2(vo1)))
        vout = self.fc3(vout)

        aout = self.relu(self.dropout2(self.fc2(ax)))
        aout = self.fc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,attx,vo1,ax  # [n, 8,2]  


#两个AM，dual-guide attention
class FNet14(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.audio_attfc2 = nn.Linear(128, 64)
        self.audio_attfc3 = nn.Linear(64, 1)
        self.visual_attfc2 = nn.Linear(128, 64)
        self.visual_attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.relu(self.dropout1(self.fc1(vx)))
        ax = self.relu(self.afc(ax))

        visual_att = self.visual_att_net(vo1,ax)
        audio_att = self.audio_att_net(ax,vo1)

        visual_att_x = vo1+visual_att
        audio_att_x = ax+audio_att


        visual_att_out = self.relu(self.dropout2(self.visual_attfc2(visual_att_x)))
        visual_att_out = self.visual_attfc3(visual_att_out)

        audio_att_out = self.relu(self.dropout2(self.audio_attfc2(audio_att_x)))
        audio_att_out = self.audio_attfc3(audio_att_out)

        vout = self.relu(self.dropout2(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.relu(self.dropout2(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return visual_att_out,audio_att_out,vout,aout,visual_att_x,audio_att_x,vo1,ax  # [n, 8,2]  

class HNet(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

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
        self.fc1 = nn.Linear(input_dim, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x [n, 8, 1024]
        # x = torch.cat((xi, xj)).view(-1, 1024)    
        out = self.relu(self.fc1(x))
        out = self.bn2(self.relu(self.fc2(out)))
        out = self.fc3(out).view(-1,args.num_per_group).contiguous()
        out = self.softmax(out)

        return out

