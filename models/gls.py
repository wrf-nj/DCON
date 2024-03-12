
import torch
from torch import nn
from torch.nn import functional as F




class GlsBlock(nn.Module):
    def __init__(self,alpha=0.1,out_channel = 32, in_channel = 3, scale_pool = [1, 3], use_act = True, requires_grad = False):
        super(GlsBlock, self).__init__()

        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        self.mixalpha=alpha
        
        assert requires_grad == False

    def forward(self, x_in):
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]] 
        nb, nc, nx, ny = x_in.shape
        ker = torch.randn([self.out_channel * nb, self.in_channel , k, k ], requires_grad = self.requires_grad).cuda()

        x_in = x_in.view(1, nb * nc, nx, ny)
        x_in = F.conv2d(x_in, ker, stride =1, padding = k //2, dilation = 1, groups = nb )
        x_in = x_in.view(nb, self.out_channel, nx, ny)

        #shift
        miu=torch.mean(x_in,axis=[2,3],keepdim=True)
        var=torch.var(x_in,axis=[2,3],keepdim=True)
        sig=(var+1e-6).sqrt()
        miu,sig=miu.detach(),sig.detach()
        
        x_in_normed=(x_in-miu)/sig
        perm=torch.randperm(nb)
            
        miu2,sig2=miu[perm],sig[perm]
        betadistribution=torch.distributions.Beta(self.mixalpha, self.mixalpha)
        lmda=betadistribution.sample((nb, 1, 1, 1))
        lmda=lmda.cuda()
            
        miu_mix=miu*lmda+miu2*(1-lmda)
        sig_mix=sig*lmda+sig2*(1-lmda)
        x_in=sig_mix*x_in_normed+miu_mix
        
        if self.use_act==True:
            x_in = F.leaky_relu(x_in)

        return x_in


class Gls(nn.Module):
    def __init__(self, alpha=1.0,glsmix_f=1,out_channel = 3, in_channel = 3, interm_channel = 2, scale_pool = [1, 3], n_layer = 4, out_norm = 'frob'):
        super(Gls, self).__init__()

        self.scale_pool = scale_pool 
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.in_channel=in_channel
        self.out_channel = out_channel
        self.interm_channel=interm_channel
        self.glsmix_f=glsmix_f
        self.mixalpha=alpha

        self.layers.append(
            GlsBlock(alpha=self.mixalpha,out_channel = self.interm_channel, in_channel = self.in_channel, scale_pool = scale_pool).cuda()
            )
        
        for ii in range(n_layer - 2):
            self.layers.append(
            GlsBlock(alpha=self.mixalpha,out_channel = self.interm_channel, in_channel = self.interm_channel, scale_pool = scale_pool).cuda()
            )

        self.layers.append(
            GlsBlock(alpha=self.mixalpha,out_channel = self.out_channel, in_channel = self.interm_channel, scale_pool = scale_pool, use_act = False).cuda()
            )
                
        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        nb, nc, nx, ny = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)

        if self.glsmix_f==1:
            mixed = alphas * x + (1.0 - alphas) * x_in
        elif self.glsmix_f==0:
            mixed=x

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed
