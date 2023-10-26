from functools import partial
from collections import OrderedDict
from sk_att import *
import torch
import torch.nn as nn
from einops import rearrange, repeat
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,  
                 qk_scale=None,     
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, frames=95, n_frame=10, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim 
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_time = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, frames+1, embed_dim))
        self.pos_embed_time_n = nn.Parameter(torch.zeros(1, n_frame, embed_dim))
        self.pos_embed_time_N = nn.Parameter(torch.zeros(1, 9, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)] 
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.blocks_time = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(2)    
        ])
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_time = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed_time_N, std=0.02)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        self.soft = nn.Softmax(dim=1)  

        self.N = n_frame
        self.confidence_threshold = 0.1
        config = [[64, 64, 16, 1], [64, 64, 16, 1],
          [64, 128, 32, 2], [128, 128, 32, 1],
          [128, 256, 64, 2], [256, 256, 64, 1],
          [256, 256, 64, 1], [256, 256, 64, 1],
          ]
        self.DSTA_net = DSTANet(config=config)  
        self.DSTA_net_body = DSTANet(config=config, num_point=67)

        self.cro_att_i = CrossAttention(dim=768, num_heads=4)
        self.cro_att_p = CrossAttention(dim=768, num_heads=4)
        
        self.cro_face_384 = nn.Linear(768, 384)
        self.cro_body_384 = nn.Linear(768, 384)

    def forward_features(self, x):
        x = self.patch_embed(x)  

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def tim_transformer(self, x):
        cls_token = self.cls_token_time.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
          
            x = torch.cat((cls_token, x), dim=1)  
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed_time_N)
        x = self.blocks_time(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])   
        else:
            return x[:, 0], x[:, 1]

    def find_apex(self, x):
        x = rearrange(self.head(x), 'b 1 c-> b (1 c)')
        x = self.soft(x)

        top_two = torch.topk(x, 2)[0]

        top_one = torch.max(top_two, dim=1)[0]
        sec_one = torch.min(top_two, dim=1)[0]
        min_score = torch.min(x, dim=1)[0]

        alph = top_one - sec_one
        beta = alph/(top_one-min_score)
        gama = alph * beta

        return gama, x

    def tim_transformer_n(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks_time(x)
        x = self.norm(x)

        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        return x

    def enhance(self, gama_fa, cro_fea_face, fa_soft):
        ga_fa = torch.stack(gama_fa, dim=1) 
        frames_face = torch.stack(cro_fea_face, dim=1)  
        frames_face = rearrange(self.cro_face_384(frames_face), 'b t 1 c-> b t (1 c)') 
        so_fa = torch.stack(fa_soft, dim=1)  

        top_face_N = torch.topk(ga_fa, 8)
        top_ga_fa_ch = torch.unsqueeze(ga_fa, 2)  
        top_enhance_face = torch.zeros([top_ga_fa_ch.shape[0], 8, 384])
        top_face_so = torch.zeros([top_ga_fa_ch.shape[0], 8, 11])

        position = top_face_N[1]
        position = torch.sort(position, dim=1, descending=False)
        for b in range(0, top_ga_fa_ch.shape[0]):
            feature_to_enhance = frames_face[b, position[0][b], :]
            ga_to_enhance = top_ga_fa_ch[b, position[0][b], :]
            ga_to_enhance = ga_to_enhance / torch.sum(ga_to_enhance, dim=0)
            top_enhance_face[b, :, :] = feature_to_enhance * (100 * ga_to_enhance)
            top_face_so[b, :, :] = so_fa[b, position[0][b], :]

        return top_enhance_face, top_face_so

    def forward(self, frames, faces, bodies, l_hand, r_hand):
        x_feature = []
        cro_fea_face, cro_fea_body = [], []
        gama_fa, gama_bo = [], []
        fa_soft, bo_soft = [], []
       
        batch = frames.shape[0]
       
        frames = rearrange(frames, 'b t c w h -> (b t) c w h')
        x = self.forward_features(frames)  

        face_x = faces[:, :, :, 0].clone()
        face_y = faces[:, :, :, 1].clone()
        face_confidences = faces[:, :, :, 2].clone()
        t = 0.1
        face_confidences = (face_confidences > t).float() * 1  
        face = torch.stack(
            (face_x * face_confidences, face_y * face_confidences), dim=3)  
        face_feature = self.DSTA_net(face)  
        body = torch.cat((bodies, l_hand, r_hand), dim=2)
        body = body.view(body.size(0), body.size(1), -1, 3)
        body_x = body[:, :, :, 0].clone()
        body_y = body[:, :, :, 1].clone()
        body_confidences = body[:, :, :, 2].clone()
        t = 0.1
        body_confidences = (body_confidences > t).float() * 1  
        body = torch.stack(
            (body_x * body_confidences, body_y * body_confidences), dim=3)
        
        body_points = self.DSTA_net_body(body) 
        x = rearrange(x, '(b t) c -> b t c', b=batch) 
        frame_list = torch.chunk(x, x.size(1), dim=1)
        for i in range(0, len(x[1])):
            frame = frame_list[i]
            cross_att_up = self.cro_att_i.forward(frame, face_feature)  
            gama_face, face_soft = self.find_apex(cross_att_up)
            fa_soft.append(face_soft)
            gama_fa.append(gama_face) 
            cro_fea_face.append(cross_att_up)  
            cross_att_down = self.cro_att_p.forward(frame, body_points)
            gama_body, body_soft = self.find_apex(cross_att_down)
            bo_soft.append(body_soft)
            gama_bo.append(gama_body)
            cro_fea_body.append(cross_att_down)

        top_enhance_face, top_face_so = self.enhance(gama_fa, cro_fea_face, fa_soft)
        top_enhance_body, top_body_so = self.enhance(gama_bo, cro_fea_body, bo_soft)

        x = torch.cat([top_enhance_face, top_enhance_body], dim=2).cuda()
        x = self.tim_transformer(x)
        x = self.head(x)

        return x, top_face_so, top_body_so


def _init_vit_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 11, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=4,
                              num_heads=4,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 kv_bias=False,
                 q_bias=False,
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.,
                 drop=0.5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim * 1, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv = nn.Conv1d(in_channels=256, out_channels=768, kernel_size=1, stride=1)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x, y):
        y = y.unsqueeze(2)
        y = self.conv(y)
        y = rearrange(y, 'b c 1 -> b 1 c')
        Bx, Nx, Cx = x.shape  
        By, Ny, Cy = y.shape   

        q = self.q(x).reshape(Bx, Nx, 1, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(By, Ny, 2, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)  
      

        q, k, v = q[0], kv[0], kv[1]  
        kv = (q @ k.transpose(-2, -1)) * self.scale

        kv = kv.softmax(dim=-1)
        kv = self.attn_drop(kv)
        kv = (kv @ v).transpose(1, 2).reshape(By, Nx, Cy)
        kv = self.proj(kv)
        kv = self.proj_drop(kv)  
        cro = x + kv

        return cro

