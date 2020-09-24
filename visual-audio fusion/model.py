# implement two networks:
# 1. f(x): ranking function
# 2. h(xi, xj): checking function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from opts import args
from torch.nn import init

class LSTM(nn.Module):
    def __init__(self, in_feature_dim=128, hidden_dim=128,num_layers=1):
        super().__init__()
        self.feature_dim = in_feature_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: tensor, (batch, seq_len, feature_dim)
        out = self.lstm(x)
        # output((batch, seq_len, num_directions*hidden_size), (h_n, c_n))
        return out[0]
#multi head,dropout
class AttentionUnit(nn.Module):
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
class AttentionModule(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        return concat

#multi head,dropout, 最后没有dropout
class AttentionUnit_1(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.dropout(self.relu(self.fc1(x)))
        output =self.relu(self.fc2(output))
        return output
class AttentionModule_1(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit_1(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        return concat

#multi head
class Fusion_AttentionUnit(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.afc = nn.Linear(in_feature_dim,64)
        self.vfc = nn.Linear(in_feature_dim,64)
        self.fusion = nn.Linear(64,out_feature_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.5)

 

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        # x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        vx = self.dropout1(self.relu(self.vfc(vx)))
        ax = self.dropout2(self.relu(self.afc(ax)))
        fx = self.relu(self.fusion(vx+ax))
        return fx
class Fusion_AttentionModule(nn.Module):
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
            self.attentionUnits.append(Fusion_AttentionUnit(feature_dim,int(feature_dim/num_of_subAttention)))
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

#multi head
class Point_AttentionUnit(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.subspace = 64
        self.Q = nn.Linear(in_feature_dim,self.subspace)
        self.W = nn.Linear(in_feature_dim,self.subspace)
        self.P = nn.Linear(self.subspace,1)

        self.V = nn.Linear(in_feature_dim,out_feature_dim)
        # self.dropout3 = nn.Dropout(0.5)

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        # x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        sigma = self.relu(self.P(self.Q(vx)+self.W(ax)))
        fx = sigma*self.V(ax)
        return fx
class Point_AttentionModule(nn.Module):
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
            self.attentionUnits.append(Point_AttentionUnit(feature_dim,int(feature_dim/num_of_subAttention)))
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


#
class Highlight_Estimation_Module(nn.Module):
    def __init__(self,**args):
        super(MILModel3, self).__init__()
        self.subspace_dim = 128
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,ffeat):
        scores = self.W(self.tanh(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        scores = F.softmax(scores,2)
        
        return scores

#一个AM，audio-guide attention
class FNet(nn.Module):
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



#一个AM，audio-guide attention
class softmax_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 2)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 2)
        self.afc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        
        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.softmax(self.attfc3(att_out))[:,1].view(-1,1).contiguous()

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.softmax(self.vfc3(vout))[:,1].view(-1,1).contiguous()

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.softmax(self.afc3(aout))[:,1].view(-1,1).contiguous()
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class attention_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.WV = nn.Linear(128, 128)
        self.WA = nn.Linear(128, 128)

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
        # att = self.sigmoid(att)
        visual_att_x = att+vo1
        audio_att_x = att+ax
        att_x = self.relu(self.WV(visual_att_x)+self.WA(audio_att_x))
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class AFNet(nn.Module):
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
        att_x = att+ax
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  


#一个AM，audio-guide attention
class lstm_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.alstm = LSTM(in_feature_dim = 128)
        self.vlstm = LSTM(in_feature_dim = 512)

        self.att_net = AM(256)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(256, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(256, 64)
        self.afc2 = nn.Linear(256, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        b,n,sq,dim = vx.shape
        vx = vx.view(b*n,sq,dim)
        vx = vx.permute(1,0,2)
        b,n,sq,dim = ax.shape
        ax = ax.view(b*n,sq,dim)
        ax = ax.permute(1,0,2)
        
        ax = self.alstm(ax)
        vx = self.vlstm(vx)

        vx = vx.permute(1,0,2)
        ax = ax.permute(1,0,2)

        vx = vx[:,int(sq/2),:]
        ax = ax[:,int(sq/2),:]
        # x [n, 8, 512]
        # x = x.view(-1, 512)

        att_x = self.att_net(vx,ax)
        # att_x = att+vx

        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vx)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vx,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class lstm_FNet1(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.alstm = LSTM(in_feature_dim = 128)
        self.vlstm = LSTM(in_feature_dim = 512)

        self.att_net = AM(256)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(256, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        # self.vfc2 = nn.Linear(256, 64)
        # self.afc2 = nn.Linear(256, 64)
        # self.vfc3 = nn.Linear(64, 1)
        # self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        b,n,sq,dim = vx.shape
        vx = vx.view(b*n,sq,dim)
        vx = vx.permute(1,0,2)
        b,n,sq,dim = ax.shape
        ax = ax.view(b*n,sq,dim)
        ax = ax.permute(1,0,2)
        
        ax = self.alstm(ax)
        vx = self.vlstm(vx)

        vx = vx.permute(1,0,2)
        ax = ax.permute(1,0,2)

        vx = vx[:,int(sq/2),:]
        ax = ax[:,int(sq/2),:]
        # x [n, 8, 512]
        # x = x.view(-1, 512)

        att_x = self.att_net(vx,ax)
        # att_x = att+vx

        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.attfc2(vx)))
        vout = self.attfc3(vout)

        aout = self.dropout2(self.relu(self.attfc2(ax)))
        aout = self.attfc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vx,ax  # [n, 8,2]  


#两个AM, dual-guide attention
class Dual_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.fusion = nn.Linear(128*2, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.visual_attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.visual_attfc3 = nn.Linear(64, 1)

        self.audio_attfc2 = nn.Linear(128, 64)
        self.audio_attfc3 = nn.Linear(64, 1)

        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.dropout6 = torch.nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax =  self.relu(self.afc(ax))

        visual_att = self.visual_att_net(vo1,ax)
        visual_att_x = visual_att+vo1
        audio_att = self.audio_att_net(ax,vo1)
        audio_att_x = audio_att+ax

        # att_x = self.dropout5(self.relu(self.fusion(torch.cat((visual_att_x,audio_att_x),1))))

        visual_att_out = self.dropout2(self.relu(self.visual_attfc2(visual_att_x)))
        visual_att_out = self.visual_attfc3(visual_att_out)

        audio_att_out = self.dropout2(self.relu(self.audio_attfc2(audio_att_x)))
        audio_att_out = self.audio_attfc3(audio_att_out)

        vout = self.dropout3(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)
        aout = self.dropout4(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return visual_att_out,audio_att_out,vout,aout,visual_att_x,audio_att_x,vo1,ax  # [n, 8,2]  

#两个AM, dual-guide attention,统一回归器
class Dual_FNet_cat(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.fusion = nn.Linear(128*2, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.dropout6 = torch.nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        visual_att = self.visual_att_net(vo1,ax)
        visual_att_x = visual_att+vo1
        audio_att = self.audio_att_net(ax,vo1)
        audio_att_x = audio_att+ax

        att_x = self.dropout5(self.relu(self.fusion(torch.cat((visual_att_x,audio_att_x),1))))

        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout3(self.relu(self.attfc2(vo1)))
        vout = self.attfc3(vout)
        aout = self.dropout4(self.relu(self.attfc2(ax)))
        aout = self.attfc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#两个AM, dual-guide attention,统一回归器
class Dual_FNet_add(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.WA = nn.Linear(128, 128)
        self.WV = nn.Linear(128, 128)

        self.fusion = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.dropout6 = torch.nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        visual_att = self.visual_att_net(vo1,ax)
        visual_att_x = visual_att+vo1
        audio_att = self.audio_att_net(ax,vo1)
        audio_att_x = audio_att+ax

        att_x = self.dropout5(self.relu(self.fusion(self.WV(visual_att_x)+self.WA(audio_att_x))))

        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout3(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)
        aout = self.dropout4(self.relu(self.afc2(ax)))
        aout = self.afc2(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  


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
        out = self.dropout1(self.relu(self.fc1(x)))
        out = self.dropout2(self.relu(self.fc2(out)))
        out = self.fc3(out).view(-1,args.num_per_group).contiguous()
        out = self.softmax(out)

        return out


class HNet2(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        self.n = n
        self.vfc = nn.Linear(args.feature_dim,args.audio_dim)
        self.afc = nn.Linear(args.audio_dim, args.audio_dim)
        self.fusion = nn.Linear(args.audio_dim*2, args.audio_dim)

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(args.audio_dim*2, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, vxi,axi,vxj,axj):
        #x [n, 8, 1024]
        # x = torch.cat((xi, xj)).view(-1, 1024)  
        vxi = self.dropout(self.relu(self.vfc(vxi)))
        vxj = self.dropout(self.relu(self.vfc(vxj)))
        axi = self.dropout(self.relu(self.afc(axi)))
        axj = self.dropout(self.relu(self.afc(axj)))

        xi = torch.cat((vxi,axi),1)
        xj = torch.cat((vxj,axj),1)
        xi = self.dropout(self.relu(self.fusion(xi)))
        xj = self.dropout(self.relu(self.fusion(xj)))

        x = torch.cat((xi,xj),1)

        out = self.dropout(self.relu(self.fc2(x)))
        out = self.fc3(out).view(-1,args.num_per_group).contiguous()
        out = self.softmax(out)

        return out

class HNet3(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        
        self.softmax = nn.Softmax()

    def forward(self, out):
        out = out.view(-1,args.num_per_group)
        argidxs = torch.argsort(-out,1)
        mid = argidxs[:,int(args.num_per_group/2)].view(-1,1)
        midv = torch.gather(out, 1, mid.long()).view(-1,1)
        boolean = out>midv
        out = self.softmax(out)
        out = out*boolean.float()
        return out

class HNet4(nn.Module):
    def __init__(self, input_dim, n=8):
        super().__init__()
        
        self.softmax = nn.Softmax(1)

    def forward(self, out):
        out = out.view(-1,args.num_per_group)
      
        out = self.softmax(out)
        return out

class MILModel(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,**args):
        super(MILModel, self).__init__()
        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leaklyrelu = nn.LeakyReLU()
    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        catfeat = torch.cat((vfeat,afeat),-1)
        ffeat = self.fusionFc(catfeat) #4096*32*128
        scores = self.W(self.sigmoid(self.V(ffeat))*self.tanh(self.U(ffeat)))
        # pdb.set_trace()
        # zfeat = torch.sum(scores*ffeat,1)
        # pdb.set_trace()
        scores = torch.transpose(scores,2, 1)  # KxN
        scores = F.softmax(scores,2)

        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.cat(zfeat,0)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)

        return scores,logits

class MILModel2(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,**args):
        super(MILModel2, self).__init__()
        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 1)
        self.V = nn.Linear(self.subspace_dim, 1)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        catfeat = torch.cat((vfeat,afeat),-1)
        ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.sigmoid(self.V(ffeat))*self.tanh(self.U(ffeat))
        # pdb.set_trace()
        ffeat = torch.sum(scores*ffeat,1)
        logits = self.classifier(ffeat)

        # logits = torch.sum(scores*logits,1)

        b,n,d = scores.shape
        scores = scores.view(b,n)
        scores = self.softmax(scores)

        return scores,logits

class MILModel3(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel3, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.tanh(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        scores = F.softmax(scores,2)

        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel3_1(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel3_1, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.sigmoid(self.V(ffeat))*self.relu(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        scores = F.softmax(scores,2)

        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel4(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel4, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.relu(self.W(self.sigmoid(self.V(ffeat))*self.tanh(self.U(ffeat))))
        # pdb.set_trace()
        scores = scores.squeeze(0)
        
        return scores

class MILModel5(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,**args):
        super(MILModel5, self).__init__()
        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leaklyrelu = nn.LeakyReLU()
    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        catfeat = torch.cat((vfeat,afeat),-1)
        ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        # feat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = (self.W(self.tanh(self.V(ffeat))*self.sigmoid(self.U(ffeat))))
        # pdb.set_trace()
        scores = scores.squeeze(0)

        return scores

class MILModel6(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel6, self).__init__()
        # self.att_net = AM(128)
        self.P = nn.Linear(64, 1)
        self.Q = nn.Linear(128, 64)
        self.K = nn.Linear(128, 64)
        self.T = nn.Linear(128, 128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        sigma = self.relu(self.P(self.Q(ax)+self.K(vo1)))
        attfeat = sigma*self.T(ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.tanh(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        scores = F.softmax(scores,2)

        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel7(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel7, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        f_v = self.relu(self.V(ffeat))
        f_u = self.relu(self.V(ffeat))

        b,n,d = f_v.shape
        sc = []
        for (fv,fu) in zip(f_v,f_u):
            eps_f = [torch.add(fv[i].expand(n, d), fu) for i in range(n)]
            eps_f = torch.stack(eps_f)
            scores = self.W(eps_f).squeeze(-1)
            scores = F.softmax(scores,0)
            scores = scores.mean(1)
            sc.append(scores)
        scores = torch.stack(sc).unsqueeze(2)
        scores = torch.transpose(scores,2, 1)  # KxN
        if self.training is True:
            scores = F.softmax(scores,2)
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel8(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel8, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN

        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits


class TaskBranch(nn.Module):
    def __init__(self,subspace_dim=128):
        super(TaskBranch, self).__init__()

        self.U = nn.Linear(subspace_dim, 64)
        self.V = nn.Linear(subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self,ffeat):

        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        if self.training is True:
            scores = F.softmax(scores,2)
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MTMILModel(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None):
        super(MTMILModel, self).__init__()
        if args.dataset == 'youtube':
            self.domains = ['dog','parkour','gymnastics','surfing','skating','skiing']
        if args.dataset == 'tvsum':
            self.domains = ['BK','BT','DS','FM','GA','MS','PK','PR','VT','VU']
        if args.dataset == 'cosum':
            self.domains = ['statue_of_liberty','eiffel_tower','NFL','kids_playing_in_leaves','MLB','excavator_river_cross','notre_dame_cathedral'
                            'surf','bike_polo','base_jump']
        
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.dropout = nn.Dropout(0.5)
        
        self.Branchs = nn.ModuleList()
        for N in range(len(self.domains)):
            self.Branchs.append(TaskBranch(self.subspace_dim))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeats,afeats,domain=None):
        # pdb.set_trace()
        if self.training:
            scores,logits = [],[]
            b,ds,n,d = vfeats.shape
            for bcidx in range(ds):
                vfeat = vfeats[:,bcidx,:,:]
                afeat = afeats[:,bcidx,:,:]
                vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
                ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
                attfeat = self.att_net(vo1,ax)
                ffeat = attfeat+vo1
                ffeat = ffeat.view(b,n,-1)
                score,logit = self.Branchs[bcidx](ffeat)
                scores.append(score)
                logits.append(logit)
        else:
            b,n,d = vfeats.shape
            vo1 = self.dropout(self.relu(self.vfc(vfeats))).view(-1,self.subspace_dim)
            ax = self.relu(self.afc(afeats)).view(-1,self.subspace_dim)
            attfeat = self.att_net(vo1,ax)
            ffeat = attfeat+vo1
            ffeat = ffeat.view(b,n,-1)
            scores,logits = self.Branchs[domain](ffeat)
        return scores,logits



class MILModel_vision(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel_vision, self).__init__()
        # self.att_net = AM(128)
        self.vfc = nn.Sequential(nn.Linear(visual_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        )
        

        # self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        ffeat = self.dropout(self.relu(self.vfc(vfeat)))
        # ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        # attfeat = self.att_net(vo1,ax)
        # ffeat = attfeat+vo1
        # ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
     
        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits


class MILModel_audio(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel_audio, self).__init__()
        # self.att_net = AM(128)
        self.afc = nn.Sequential(nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        )
        

        # self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        # ffeat = self.dropout(self.relu(self.vfc(vfeat)))
        ffeat = self.relu(self.afc(afeat))
        
        # attfeat = self.att_net(vo1,ax)
        # ffeat = attfeat+vo1
        # ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
     
        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits
# score without softmax
class MILModel9(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel9, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        scores_noso = scores
        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores_noso = scores_noso.squeeze(1)
        scores = scores.squeeze(1)
        return scores_noso,logits

class MILModel10(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel10, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        # self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        
        # if self.training is True:
        scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel11(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel11, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        # self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        initScores = self.W(self.relu(self.V(ffeat)))
        initScores = torch.transpose(initScores,2, 1)  # KxN
        
        if self.training is True:
            sigmoidScores = F.sigmoid(initScores)
            sigmoidScores = sigmoidScores.squeeze(1)
        weights = F.softmax(initScores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(weights,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        if self.training is True:
            return sigmoidScores,logits
        else:
            return initScores,logits


class MILModel12(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel12, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        # self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        initScores = self.W(self.relu(self.V(ffeat)))
        initScores = torch.transpose(initScores,2, 1)  # KxN
        
        
        weights = F.softmax(initScores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(weights,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        initScores = initScores.squeeze(1)

        return initScores,logits

class Gated_Attention(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(Gated_Attention, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.tanh(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN

        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits
