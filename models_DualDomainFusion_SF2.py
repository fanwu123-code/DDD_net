import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
from einops import rearrange

from torch import Tensor
from typing import Optional

from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)
        
        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                           kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        
        self.freq_attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels//2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels//2, out_channels=2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.fpe_small = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                               padding=1, stride=1, groups=in_channels, bias=True)
        self.fpe_large = nn.Conv2d(in_channels * 2, in_channels, kernel_size=5,
                               padding=2, stride=1, groups=in_channels, bias=True)
        
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1),
            nn.GELU()
        )
        
        self.low_freq_enhance = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        
        ffted = self.bn(ffted)
        
        fpe_small = self.fpe_small(ffted)
        fpe_large = self.fpe_large(ffted)
        ffted = ffted + torch.cat([fpe_small, fpe_large], dim=1)
        
        freq_weights = self.freq_attention(ffted)
        
        mask_h = torch.ones((h, w//2+1), device=x.device)
        center_h, center_w = h//2, (w//2+1)//2
        for i in range(h):
            for j in range(w//2+1):
                if ((i-center_h)**2 + (j-center_w)**2) < min(h, w)//4:
                    mask_h[i, j] = 0
        
        high_freq = ffted * mask_h.reshape(1, 1, h, w//2+1)
        high_freq = self.high_freq_enhance(high_freq)
        low_freq = ffted * (1 - mask_h).reshape(1, 1, h, w//2+1)
        low_freq = self.low_freq_enhance(low_freq)
        
        ffted = freq_weights[:, 0:1, :, :] * low_freq + freq_weights[:, 1:2, :, :] * high_freq
        
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2*c, h, -1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        
        return output

class TokenMixer_For_Local(nn.Module):
    def __init__(self, dim):
        super(TokenMixer_For_Local, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,stride=1)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,stride=1)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,stride=1)
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.gelu(x1+x)

        x2 = self.conv2(x1)
        x2 = self.gelu(x2+x1+x)

        x3 = self.conv3(x2)
        x3 = self.gelu(x3+x2+x1+x)

        x4 = self.conv4(x3)
        x4 = self.gelu(x4+x3+x2+x1+x)

        x5 = self.conv5(x4)
        x5 = self.gelu(x5+x4+x3+x2+x1+x)

        x6= self.conv6(x5)
        x6 = self.gelu(x6+x5+x4+x3+x2+x1+x)

        return x6

class TokenMixer_For_Global(nn.Module):
    def __init__(self, dim):
        super(TokenMixer_For_Global, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)

        return x

class CrossTransformer(nn.Module):
    def __init__(self, dropout, d_model=512, n_head=4):
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        dif = input2 - input1
        output_1 = self.cross(input1, dif)
        output_2 = self.cross(input2, dif)
        cat_s = input1 + input2
        output_11 = self.cross(output_1, cat_s)
        output_22 = self.cross(output_2, cat_s)

        return output_11,output_22
    
    def cross(self, input,dif):
        attn_output, attn_weight = self.attention(input, dif, dif)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head=8, dropout=0.):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        batch, c, h, w = x.size()
        
        x_seq = x.view(batch, c, -1).permute(2, 0, 1)
        
        attn_output, _ = self.self_attn(x_seq, x_seq, x_seq)
        x_seq = x_seq + self.dropout1(attn_output)
        x_seq = self.norm1(x_seq)
        
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(x_seq))))
        x_seq = x_seq + self.dropout3(ff_output)
        x_seq = self.norm2(x_seq)
        
        output = x_seq.permute(1, 2, 0).view(batch, c, h, w)
        
        return output

class resblock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3,stride=1,padding=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)

class DualDomainTransformer(nn.Module):
    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        super(DualDomainTransformer, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.h = h
        self.w = w
        
        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h*w, int(d_model))
        
        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)
        
        self.local_processor = nn.ModuleList([TokenMixer_For_Local(d_model) for i in range(n_layers)])
        self.global_processor = nn.ModuleList([TokenMixer_For_Global(d_model) for i in range(n_layers)])
        
        self.spatial_transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        self.frequency_transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        
        self.ca_conv = nn.Sequential(
            nn.Conv2d(4*d_model, 2*d_model, 1),
        )
        
        self.domain_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model*4, d_model*2, 1),
                nn.GELU(),
                nn.Conv2d(d_model*2, d_model*4, 1)
            ) for i in range(n_layers)
        ])
        
        self.ca = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(d_model*4, d_model*2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model*2, d_model*4, kernel_size=1),
                nn.Sigmoid()
            ) for i in range(n_layers)
        ])
        
        self.resblock = nn.ModuleList([resblock(d_model*2, d_model*2) for i in range(n_layers)])
        self.LN = nn.ModuleList([nn.LayerNorm(d_model*2) for i in range(n_layers)])
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
                
    def forward(self, img_feat1, img_feat2):
        batch = img_feat1[0].size(0)
        feature_dim = img_feat1[0].size(1)
        h, w = self.h, self.w
        
        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                      embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                     dim=-1)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        
        fused_features_list = []
        
        for i in range(self.n_layers):
            if feature_dim == 1024:
                img_feat1_scale = self.projection(img_feat1[i])  
                img_feat2_scale = self.projection(img_feat2[i])
            elif feature_dim == 768:
                img_feat1_scale = self.projection2(img_feat1[i])  
                img_feat2_scale = self.projection2(img_feat2[i])
            elif feature_dim == 512:
                img_feat1_scale = self.projection3(img_feat1[i])  
                img_feat2_scale = self.projection3(img_feat2[i])
            elif feature_dim == 256:
                img_feat1_scale = self.projection4(img_feat1[i])  
                img_feat2_scale = self.projection4(img_feat2[i])
            
            spatial_feat1 = self.local_processor[i](img_feat1_scale)
            frequency_feat1 = self.global_processor[i](img_feat1_scale)
            spatial_feat2 = self.local_processor[i](img_feat2_scale)
            frequency_feat2 = self.global_processor[i](img_feat2_scale)
            
            spatial_feat1 = spatial_feat1 + position_embedding
            spatial_feat2 = spatial_feat2 + position_embedding
            frequency_feat1 = frequency_feat1 + position_embedding
            frequency_feat2 = frequency_feat2 + position_embedding
            
            spatial_seq1 = spatial_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)
            spatial_seq2 = spatial_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)
            spatial_out1, spatial_out2 = self.spatial_transformer[i](spatial_seq1, spatial_seq2)
            
            freq_seq1 = frequency_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)
            freq_seq2 = frequency_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)
            freq_out1, freq_out2 = self.frequency_transformer[i](freq_seq1, freq_seq2)
            
            spatial_out1 = spatial_out1.permute(1, 2, 0).view(batch, self.d_model, h, w)
            spatial_out2 = spatial_out2.permute(1, 2, 0).view(batch, self.d_model, h, w)
            freq_out1 = freq_out1.permute(1, 2, 0).view(batch, self.d_model, h, w)
            freq_out2 = freq_out2.permute(1, 2, 0).view(batch, self.d_model, h, w)
            
            spatial_out = torch.cat([spatial_out1, spatial_out2], dim=1)
            freq_out = torch.cat([freq_out1, freq_out2], dim=1)
            combined_features = torch.cat([spatial_out, freq_out], dim=1)
            
            fused_features = self.domain_fusion[i](combined_features) + combined_features
            fused_features = self.ca[i](fused_features) * fused_features
            fused_features = self.ca_conv(fused_features)
            fused_features_list.append(fused_features)
            
        i = 0
        output = torch.zeros((h*w, batch, self.d_model*2)).to(device)
        for res in self.resblock:
            input = fused_features_list[i]
            output = output.permute(1, 2, 0).view(batch, self.d_model*2, h, w) + input
            output = res(output)
            output = output.view(batch, self.d_model*2, -1).permute(2, 0, 1)
            output = self.LN[i](output)
            i = i + 1
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        enc_att, att_weight = self._mha_block2((self_att_tgt),
                                               memory, memory_mask,
                                               memory_key_padding_mask)
     
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))

        return x
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)


    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x,att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),att_weight
    def _mha_block2(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn2(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout3(x),att_weight
    def _mha_block3(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn3(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x),att_weight


    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)



class DecoderTransformer(nn.Module):
    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
        super(DecoderTransformer, self).__init__()

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)

        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim)

        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()

    def init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, memory, encoded_captions, caption_lengths):
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)

        pred = self.transformer(tgt_embedding, memory, tgt_mask=mask)
        pred = self.wdc(self.dropout(pred))

        pred = pred.permute(1, 0, 2)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind

