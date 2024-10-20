import torch.nn as nn
import torch.nn.functional as F

NetParameter = {
    'nInput':300,
    'nEmb':256,
    'nFw':512,
    'nAttnHead':4,
    'nLayer':2
}


class KoiKoiEncoderBlock(nn.Module):
    def __init__(self, nInput, nEmb, nFw, nAttnHead, nLayer):
        super(KoiKoiEncoderBlock,self).__init__()
        self.f1 = nn.Conv1d(nInput, nFw, 1)
        self.f2 = nn.Conv1d(nFw, nEmb, 1)
        attn_layer = nn.TransformerEncoderLayer(nEmb, nAttnHead, nFw)
        self.attn_encoder = nn.TransformerEncoder(attn_layer, nLayer)
        
    def forward(self,x): 
        x = self.f2(F.relu(self.f1(x)))
        x = F.layer_norm(x,[x.size(-1)])
        x = x.permute(2,0,1)
        print("after permute")
        print(x)
        x = self.attn_encoder(x)
        print("after attn encoder")
        print(x)
        x = x.permute(1,2,0) 
        print(x) 
        return x


class DiscardModel(nn.Module):
    def __init__(self):
        super(DiscardModel,self).__init__()       
        self.encoder_block = KoiKoiEncoderBlock(**NetParameter)
        self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self,x):       
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        print("after out")
        print(x)
        return x