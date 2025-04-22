"""
Created on Sun Apr 19 23:11:35 2020

@author: esat
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
#import torch.fft as afft

from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

from .r2plus1d import r2plus1d_34_32_ig65m, r2plus1d_34_32_kinetics, flow_r2plus1d_34_32_ig65m

from .representation_flow import resnet_50_rep_flow
#from NonLocalBlock1D import NonLocalBlock1D

__all__ = ['rgb_r2plus1d_32f_34', 'rgb_r2plus1d_32f_34_bert10', 'rgb_r2plus1d_64f_34_bert10']

class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.
    Args:
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)

        fft_product = fft1 * fft2

        cbp_flat = afft.ifft(fft_product).real

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


class rgb_r2plus1d_32f_34(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34, self).__init__()
        self.num_classes=num_classes
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.fc_action = nn.Linear(512, num_classes)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(512,256)
        for param in self.features.parameters():
            param.requires_grad = True
        print(sum(p1.numel() for p1 in self.features.parameters() if p1.requires_grad))                        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.output_dim = 512
        output_dim=512
	#self.sum_pool = sum_pool
        self.T = 4
        self.spat_x = 7
        self.spat_y = 7
        input_dim1 = 512
        input_dim2=512
        rand_h_1 =  torch.randint(output_dim, size = (input_dim1,)).repeat(self.T,1).transpose(0,1).contiguous().view(-1)
        rand_h_2 = torch.randint(output_dim, size = (input_dim2,)).repeat(self.T,1).transpose(0,1).contiguous().view(-1)
        #print(rand_h_1.shape)
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
        self.sketch_matrix1 = torch.nn.Parameter(generate_sketch_matrix(rand_h_1, 2 * torch.randint(2, size = (input_dim1*self.T,)) - 1, input_dim1*self.T, output_dim))
        self.sketch_matrix2 = torch.nn.Parameter(generate_sketch_matrix(rand_h_2, 2 * torch.randint(2, size = (input_dim2*self.T,)) - 1, input_dim2*self.T, output_dim))

        
    def forward(self, x):
        x = self.features(x)
        x1  = F.avg_pool3d(x, kernel_size=[1, self.spat_x, self.spat_y], stride=[1,1,1]).view(x.shape[0],-1)
        x2  = F.avg_pool3d(x, kernel_size=[1, self.spat_x, self.spat_y], stride=[1,1,1]).view(x.shape[0],-1)

        fft1 = torch.rfft(x1.matmul(self.sketch_matrix1), 1)
        fft2 = torch.rfft(x2.matmul(self.sketch_matrix2), 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        out = torch.irfft(fft_product, 1, signal_sizes = (self.output_dim,)) * self.output_dim	
	# signed sqrt, L2 normalization
        out = torch.mul(torch.sign(out),torch.sqrt(torch.abs(out)+1e-12))  # signed sqrt
        out = F.normalize(out, p=2, dim=1)
        print(out.shape)
        x = self.fc_action(out)
        print(x.shape)
        return x


#    def forward(self, x):
#        x = self.features(x)
#        print(x.shape)
#        x1 = x[:,:,0,:,:]
#        x2 = x[:,:,1,:,:]
#        x3 = x[:,:,2,:,:]
#        x4 = x[:,:,3,:,:]
#        print("x1",x1.shape)
#        l = CompactBilinearPooling(512, 512, 512)
#        l.cuda()
#        l.train()
#        o1 = l(x1,x1)
#        o2 = l(x2,x2)
#        o3 = l(x3,x3)
#        o4 = l(x4,x4)
#        print("o1",o1.shape)
#        oo1 = torch.reshape(o1,(4,1,512))
#        oo2 = torch.reshape(o2,(4,1,512))
#        oo3 = torch.reshape(o3,(4,1,512))
#        oo4 = torch.reshape(o4,(4,1,512))
#        print("oo",oo1.shape)
#        o = torch.cat((oo1,oo2,oo3,oo4),1)
#        print("o",o.shape)
#        #x=x.permute(0,2,1,3,4)
#        #print(.shape[1])
#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.dp(x)
##        x1  = F.avg_pool3d(x, kernel_size=[1, self.spat_x, self.spat_y], stride=[1,1,1]).view(x.shape[0],-1)
##        x2  = F.avg_pool3d(x, kernel_size=[1, self.spat_x, self.spat_y], stride=[1,1,1]).view(x.shape[0],-1)
##        #print(x1.shape)
##        #print((self.sketch_matrix1).shape)
##        fft1 = torch.rfft(x1.matmul(self.sketch_matrix1), 1)
##        fft2 = torch.rfft(x2.matmul(self.sketch_matrix2), 1)
##        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
##        out = torch.irfft(fft_product, 1, signal_sizes = (self.output_dim,)) * self.output_dim	
##	# signed sqrt, L2 normalization
##        out = torch.mul(torch.sign(out),torch.sqrt(torch.abs(out)+1e-12))  # signed sqrt
##        out = F.normalize(out, p=2, dim=1)
##        #print(out.shape)
#        x = self.fc_action(x)
#        return x

#    def forward(self, x):
#        x = self.features(x)
#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        return x
    
class rgb_r2plus1d_32f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=4
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 4 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
#    def forward(self, x):
#        x = self.features(x)
#        #print(x.shape)
#        x1 = x[:,:,0,:,:]
#        x2 = x[:,:,1,:,:]
#        x3 = x[:,:,2,:,:]
#        x4 = x[:,:,3,:,:]
##        print("x1",x1.shape)
#        l = CompactBilinearPooling(512, 512, 256)
#        l.cuda()
#        l.train()
#        o1 = l(x1,x1)
#        o2 = l(x2,x2)
#        o3 = l(x3,x3)
#        o4 = l(x4,x4)
#        #print("o1",o1.shape)
#        bs  = o1.shape[0]
#        #print("bs",bs)
#        oo1 = torch.reshape(o1,(bs,1,256))
#        oo2 = torch.reshape(o2,(bs,1,256))
#        oo3 = torch.reshape(o3,(bs,1,256))
#        oo4 = torch.reshape(o4,(bs,1,256))
#        print("oo",oo1.shape)
#        o = torch.cat((oo1,oo2,oo3,oo4),1)
#        print("o",o.shape)
#        input_vectors=o
#        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
#        input_vectors = input_vectors.div(norm)
##        output , maskSample = self.bert(x)
#        output , maskSample = self.bert(o)
#        classificationOut = output[:,0,:]
#        sequenceOut=output[:,1:,:]
#        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
#        sequenceOut = sequenceOut.div(norm)
#        output=self.dp(classificationOut)
#        x_out = self.fc_action(output)
#        return x_out, input_vectors, sequenceOut, maskSample


    def forward(self, x):
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        x = self.avgpool(x)
        print("in",x.shape)
        x = x.view(x.size(0), self.hidden_size, 4)
        x = x.transpose(1,2)
        print("re",x.shape)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class rgb_r2plus1d_64f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_64f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 8)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
    
    
