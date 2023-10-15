"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
from sk_att import *
import torch
import torch.nn as nn
from einops import rearrange, repeat
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# ****************************************
# 1.定义dropout,批嵌入，自注意，MLP,Encoder block层
# 2.定义ViT模型
# ****************************************


def drop_path(x, drop_prob: float = 0., training: bool = False):
    # 随机深度
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
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
    """
    2D Image to Patch Embedding
    卷积，展平
    """
    # norm_layer默认为None，若有传入则使用传入数据
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # grid_size 14
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 计算patches数目 14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 卷积核大小为16*16，步长16
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    # 正向传播过程： x 通过Patch Embedding
    # ViT模型中输入图片大小必须固定
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten展平处理: [B, C, H, W] -> [B, C, HW]
        # transpose调换维度上数据: [B, C, HW] -> [B, HW, C]
        # 展平从第二个维度，高开始；
        # 调换1，2位置上数据，为了实现【num_token,token_dim】
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的维数
                 num_heads=8,
                 qkv_bias=False,    # 使用qkv时是否使用偏置
                 qk_scale=None,     #
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 对每一个head的dim
        head_dim = dim // num_heads
        # 如果传入了qk_scale,就用传入的，否则使用每个头维度乘-0.5次方，对应attention公式中（QKT/根号d）*V的分母
        self.scale = qk_scale or head_dim ** -0.5
        # 使用一个全连接层直接得到qkv，全连接层节点个数*3=使用3个节点个数为dim的全连接层
        # 此处这样做的目的可能是为了并行化
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # todo dropout，训练时常用，可以避免过拟合，增强泛化能力
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 再定义一个全连接层，对应多头自注意输出d Concate拼接后，乘的W0 。输入输出节点个数都等于dim
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # 正向传播： x 通过多头自注意
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # batch：训练时，一批数据传入的小块图片
        # patches 14*14=196个+1 （加入class token），可以理解为从卷积出来的图片中的一个块，每个卷积核（过滤器）扫描的地方称为一个patch
        # embed_dim总数：经过embedding后的维数 = 768
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        #   将输入的初始qkv通过前面定义的对qkv的全连接操作得到qkv
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        #   将3*total_embed_dim拆成三个系数,3=qkv三个参数，num_heads使用heads的数目， C // self.num_heads 针对每一个heads,它的qkv对应的维度
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        #   为了方便后面做运算，
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 通过切片的方式拿到qkv的数据
        #  ####因为前面已经将数据按每个heads进行划分了，所以这里的操作均是对每个heads的qkv进行操作的
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        #   ####transpose后最后两个互换位置变成上面这样
        # @: multiply（矩阵乘法） -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        #   后面两项做矩阵乘法，一个shape=a*b,一个shape=b*a，相乘结果为a*a
        #   再乘上一个scale，就是对结果进行一个norm处理:QKT/根号d
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # dim=-1，就是对矩阵的每一行进行一个softmax处理
        attn = attn.softmax(dim=-1)
        # 对得到的结果，也就是V的权重进行dropout
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        #   前面softmax结果与V进行矩阵相乘
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        #   将最后两个维度信息拼接在一起，相当于将每个heads对应信息拼接
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        x = self.proj(x)
        # 在进行一次dropout得到最终输出
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    # （激活函数默认GELU）
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 若传入了out_features（格式）,就用传入的，否则就用in_feature
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #  第一层全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 再定义一个Dropout层
        self.drop = nn.Dropout(drop)

    # 正向传播过程: x 通过MLP Block
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Encoder Block， Transformer Encoder就是把Encoder Block 重复堆叠L次
class Block(nn.Module):
    def __init__(self,
                 # 针对每个token的维度
                 dim,
                 num_heads,
                 # 第一个全连接层节点个数是输入节点个数的四倍
                 mlp_ratio=4.,
                 qkv_bias=False,
                 # 默认为None,对应前面q*kt *的self.scale
                 qk_scale=None,
                 # 对应多头自注意模块里的drop_ratio最后全连接层后使用的drop_ratio
                 drop_ratio=0.,
                 # 对应q*kt /根号d之后通过softmax之后的dropout层
                 attn_drop_ratio=0.,
                 # 对应从多头自注意出来后的,和从MLP Block出来后的DropPath
                 drop_path_ratio=0.,
                 # 激活函数默认GELU
                 act_layer=nn.GELU,
                 # 标准化默认采用LayerNorm
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # norm1对应第一个LayerNorm
        self.norm1 = norm_layer(dim)
        #   调用前面定义的Attention类，实例化多头自注意模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        #   NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #   传入的drop_path大于0，实例化DropPath方法；否则，使用nn.Identity()，即不做任何操作。
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # norm2对应第二个LayerNorm
        self.norm2 = norm_layer(dim)
        # norm2对应第二个LayerNorm
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 初始化MLP结构（维度，隐藏层维数，激活函数，Drop_ratio）
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    # 正向传播过程: x 通过Encoder Block
    def forward(self, x):
        # x经过第一个LayerNorm、多头自注意，Drop,得到的再加上x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x经过第二个LayerNorm，MLP,Drop,得到的再加上x
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, frames=95, n_frame=10, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head  为多头自注意中heads的个数
            embed_dim (int): embedding dimension
            depth (int): depth of transformer   为在Transformer中 Transformer Encoder重复堆叠的次数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
                对应MLP Head中President-Logits中全连接层的节点个数；=None表示MLP中只有Linear一个全连接层，没有President-Logits
            distilled (bool): model includes a distillation token and head as in DeiT models 搭建DIT模型时使用
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        # 用partial（）方法为LayerNorm传入默认参数eps
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 用nn.Parameter创建可训练参数
        # cls_token 1*768,第一个1为patch长度，为了后面方便拼接；第二个1对应1*768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_time = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 可以不看用于DIT模型
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # position embedding 的shape与Concat后的shape一样，所以：
        #   1. 使用0矩阵初始化
        #   2. 1对应patch，可以不管
        #   3. num_patches=14*14=196 + token :1 = 197
        #   4. embid_dim 为传入的embid_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 时间位置嵌入
        self.pos_embed_time = nn.Parameter(torch.zeros(1, frames+1, embed_dim))
        self.pos_embed_time_n = nn.Parameter(torch.zeros(1, n_frame, embed_dim))
        self.pos_embed_time_N = nn.Parameter(torch.zeros(1, 9, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 构建等差序列，范围从0-drop_path_ratio,默认为0，使用时传入。序列中总共有depth个元素
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # Transformer Encoder 的Block
        #   1.创建列表
        #   2.for循环，重复depth次
        #   3.对每一个Block,使用的 drop_path_ratio是递增的
        #   4.其他参数不变
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
            for i in range(2)     # TODO 可修改此处参数
        ])
        # Block结束后从Transformer Encoder 出来后的标准化
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # Transformer中默认distilled等于False的，所以后面默认为真，用and连接，只看前面就行
        # 判断representation_size 是否为None，若为None,则不进行MLP中的Pre-Logits,否则，多做一个全连接层和一个Tanh()函数
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            # 有序字典
            self.pre_logits = nn.Sequential(OrderedDict([
                # 全连接层（输入节点个数，输出节点个数）
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        # 若representation_size为False,则进入这部分,对Pre-Logits不做处理
        else:
            self.has_logits = False
            # todo nn.Identity()不做任何处理
            self.pre_logits = nn.Identity()

        # Classifier head(s) 即MLP heads 里的Linear全连接层
        # 输入节点个数，输出flag类别个数
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_time = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 与ViT模型无关
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init 权重初始化
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # 时间位置权重初始化
        # nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
        # nn.init.trunc_normal_(self.pos_embed_time_n, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_time_N, std=0.02)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 时间类别token初始化权重
        # nn.init.trunc_normal_(self.cls_token_time, std=0.02)
        # self.apply(_init_vit_weights)
        self.soft = nn.Softmax(dim=1)   # softmax

        self.N = n_frame
        self.confidence_threshold = 0.1
        config = [[64, 64, 16, 1], [64, 64, 16, 1],
          [64, 128, 32, 2], [128, 128, 32, 1],
          [128, 256, 64, 2], [256, 256, 64, 1],
          [256, 256, 64, 1], [256, 256, 64, 1],
          ]
        self.DSTA_net = DSTANet(config=config)#  .cuda()
        self.DSTA_net_body = DSTANet(config=config, num_point=67)# .cuda()

        self.cro_att_i = CrossAttention(dim=768, num_heads=4)
        self.cro_att_p = CrossAttention(dim=768, num_heads=4)
        
        self.cro_face_384 = nn.Linear(768, 384)
        self.cro_body_384 = nn.Linear(768, 384)

    def forward_features(self, x):
        # [B, 1, C, H, W] -> [B, num_patches, embed_dim]
        # 对应patch embedding结构
        # x = rearrange(x, 'b 1 c w h -> (b 1) c w h')
        x = self.patch_embed(x)  # [B, 196, 768]

        # 在0这个维度上扩充成Batch size
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 在ViT中为None
        if self.dist_token is None:
            # 将cls_token和x进行拼接（在1这个维度）
            # [B, 1, 768]和[B, 196, 768]--> [B, 197, 768]
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 加上位置信息，再通过dropout层
        x = self.pos_drop(x + self.pos_embed)
        # 堆叠的一系列Encoder block
        x = self.blocks(x)
        # 对应Encoder出来的Layer norm
        x = self.norm(x)
        # 提取cls_token对应的输出
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def tim_transformer(self, x):
        # 在0这个维度上扩充成Batch size
        cls_token = self.cls_token_time.expand(x.shape[0], -1, -1)
        # 在ViT中为None
        if self.dist_token is None:
            # [B, 1, 768]和[B, T, 768]--> [B, T+1, 768]
            # print(cls_token.device)
            # print(x.device)
            x = torch.cat((cls_token, x), dim=1)  # [B, T+1, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 加上位置信息，再通过dropout层
        x = self.pos_drop(x + self.pos_embed_time_N)
        # 堆叠的一系列Encoder block
        x = self.blocks_time(x)
        # 对应Encoder出来的Layer norm
        x = self.norm(x)
        # 提取cls_token对应的输出
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])   # 时序输出
        else:
            return x[:, 0], x[:, 1]

    def find_apex(self, x):
        # x->(batch, 7)
        x = rearrange(self.head(x), 'b 1 c-> b (1 c)')
        x = self.soft(x)

        top_two = torch.topk(x, 2)[0]
        """
        1.得到各帧softmax分数x
        2.找到最大值，次大值，最小值计算显著度
        """
        top_one = torch.max(top_two, dim=1)[0]
        sec_one = torch.min(top_two, dim=1)[0]
        min_score = torch.min(x, dim=1)[0]

        alph = top_one - sec_one
        beta = alph/(top_one-min_score)
        gama = alph * beta
        # for b in range(x.shape[0]):
        #     if torch.max(top_two, dim=1)[1][b] == 4:
        #         gama[b] = gama[b]/2
        return gama, x

    def tim_transformer_n(self, x):
        # [B, N, D] -> [B, first N frames, embed_dim]
        # 在0这个维度上扩充成Batch size
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # 加上位置信息，再通过dropout层
        x = self.pos_drop(x + self.pos_embed)
        # 堆叠的一系列Encoder block
        x = self.blocks_time(x)
        # 对应Encoder出来的Layer norm
        x = self.norm(x)
        # 提取cls_token对应的输出

        # 在ViT中为None
        if self.dist_token is None:
            # 将cls_token和x进行拼接（在1这个维度）
            # [B, 1, 768]和[B, 196, 768]--> [B, 197, 768]
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 加上位置信息，再通过dropout层
        x = self.pos_drop(x + self.pos_embed)
        # 堆叠的一系列Encoder block
        x = self.blocks(x)
        # 对应Encoder出来的Layer norm
        x = self.norm(x)
        # 提取cls_token对应的输出
        if self.dist_token is None:
            # 在ViT中为None
            # 取所有batch上的，第二个维度为0的数据，因为前面x=(cls_token,x)+ self.pos_embed，
            #   所以取的值为cls_token+ pos_embed第二维度数值
            # 将切片出来的数据通过pre_logits（），前面representation_size()=0时不做pre_logits,
            #   所以这里直接返回self.cls_token对应的输出
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        return x

    def enhance(self, gama_fa, cro_fea_face, fa_soft):
        ga_fa = torch.stack(gama_fa, dim=1)  # (b, 95)
        frames_face = torch.stack(cro_fea_face, dim=1)  # (b, 95, 1, 768)
        frames_face = rearrange(self.cro_face_384(frames_face), 'b t 1 c-> b t (1 c)')  # (b, 95, 1, 384)
        so_fa = torch.stack(fa_soft, dim=1)  # (b, 95, 11)

        top_face_N = torch.topk(ga_fa, 8)
        top_ga_fa_ch = torch.unsqueeze(ga_fa, 2)  # (b, 95, 1)
        # for b in range(0, top_ga_fa_ch.shape[0]):
        # top_face = torch.zeros([top_ga_fa_ch.shape[0], 8, 384])
        top_enhance_face = torch.zeros([top_ga_fa_ch.shape[0], 8, 384])
        # top_face_ga = torch.zeros([top_ga_fa_ch.shape[0], 8, 1])
        top_face_so = torch.zeros([top_ga_fa_ch.shape[0], 8, 11])

        # top_face_list, top_face_so_list = [], []
        position = top_face_N[1]
        # TODO: 1.从小到大将最显著帧排序 √   2. 分别取对应特征 √，对应特征*对应显著度
        position = torch.sort(position, dim=1, descending=False)
        for b in range(0, top_ga_fa_ch.shape[0]):
            feature_to_enhance = frames_face[b, position[0][b], :]
            ga_to_enhance = top_ga_fa_ch[b, position[0][b], :]
            ga_to_enhance = ga_to_enhance / torch.sum(ga_to_enhance, dim=0)
            top_enhance_face[b, :, :] = feature_to_enhance * (0.5 + ga_to_enhance)
            top_face_so[b, :, :] = so_fa[b, position[0][b], :]

        return top_enhance_face, top_face_so

    def forward(self, frames, faces, bodies, l_hand, r_hand):
        # images:(b, T, 3, 224, 224)
        x_feature = []
        cro_fea_face, cro_fea_body = [], []
        gama_fa, gama_bo = [], []
        fa_soft, bo_soft = [], []
        """
        获取各帧空间特征
        1. 将堆叠的各帧图片->空间vit  (batch, frames, 768)   √
        2. 将表情关键点，姿态骨骼点通过DSTA获得特征 (batch, 1, 256)--》 (batch, frames, 768)    √
        3. 跨模态引导  --》 (batch, 1, 768)
        4. 分类->得到各帧特征和各帧分数
        5. 计算各帧显著度
        5. N_feature = 保留最大N帧分数对应的[（batch, 7）*N ]特征
        6. 为各帧加权
        7. feature = 取加权后特征进行分类(batch, 7)
        8. output = N_feature, feature 计算损失
        """
        # points_list = torch.chunk(faces, faces.size(1), dim=1)
        batch = frames.shape[0]
        # ================images=======================
        frames = rearrange(frames, 'b t c w h -> (b t) c w h')
        x = self.forward_features(frames)   # (b*95, 768)
        # =================points_face=================      # Tensor(b, 95, 70, 3)
        face_x = faces[:, :, :, 0].clone()  # (b,95, 70)
        face_y = faces[:, :, :, 1].clone()
        face_confidences = faces[:, :, :, 2].clone()
        # todo 关节置信度阈值，设为 0.1, 修改为0.3
        # t = torch.Tensor([self.confidence_threshold]).cuda()  # threshold for confidence of joints TODO .cuda()
        # 通过阈值的值变为1，否则变为0
        t = 0.1
        face_confidences = (face_confidences > t).float() * 1  # TODO .float()去掉
        # make all joints with threshold lower than
        # features_positions :Tensor(b, 95, 70, 3)
        face = torch.stack(
            (face_x * face_confidences, face_y * face_confidences), dim=3)  # (b, 95, 70, 2)

        face_feature = self.DSTA_net(face)    # (b, 256)
        # ****************points_body********************
        body = torch.cat((bodies, l_hand, r_hand), dim=2)
        body = body.view(body.size(0), body.size(1), -1, 3)
        body_x = body[:, :, :, 0].clone()
        body_y = body[:, :, :, 1].clone()
        body_confidences = body[:, :, :, 2].clone()
        # todo 关节置信度阈值，设为 0.1, 修改为0.3
        # t = torch.Tensor([self.confidence_threshold]).cuda()  # threshold for confidence of joints TODO .cuda()
        # 通过阈值的值变为1，否则变为0
        t = 0.1
        body_confidences = (body_confidences > t).float() * 1  # TODO .float()去掉
        # make all joints with threshold lower than
        # features_positions :Tensor(b, 95, 70, 3)
        body = torch.stack(
            (body_x * body_confidences, body_y * body_confidences), dim=3)
        
        body_points = self.DSTA_net_body(body)  # (b, 256)
        # ===========================================================
        x = rearrange(x, '(b t) c -> b t c', b=batch)  # 新增  (b, 95, 768)
        frame_list = torch.chunk(x, x.size(1), dim=1)
        for i in range(0, len(x[1])):
            frame = frame_list[i]
            # =================face分支==========================
            cross_att_up = self.cro_att_i.forward(frame, face_feature)   # (b, 1, 768)
            gama_face, face_soft = self.find_apex(cross_att_up)
            fa_soft.append(face_soft)
            gama_fa.append(gama_face)   # 各帧表情伽马
            cro_fea_face.append(cross_att_up)   # 各帧表情特征
            # =================body分支==========================
            cross_att_down = self.cro_att_p.forward(frame, body_points)
            gama_body, body_soft = self.find_apex(cross_att_down)
            bo_soft.append(body_soft)
            gama_bo.append(gama_body)
            cro_fea_body.append(cross_att_down)

        # top_enhance_face, top_enhance_body用来cat进行时序处理；top_face_so,top_body_so(baatch, 8, 11) 用来累加计算损失
        top_enhance_face, top_face_so = self.enhance(gama_fa, cro_fea_face, fa_soft)
        top_enhance_body, top_body_so = self.enhance(gama_bo, cro_fea_body, bo_soft)

        x = torch.cat([top_enhance_face, top_enhance_body], dim=2).cuda()
        x = self.tim_transformer(x)
        x = self.head(x)

        return x, top_face_so, top_body_so


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
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

# 构建ViT模型
# 建议使用预训练权重模型
def vit_base_patch16_224_in21k(num_classes: int = 11, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    # 调用VisionTransformer（）这个类，根据不同的配置传入不同的参数
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              # 对应图片里的Hidden_size
                              embed_dim=768,
                              # 对应图片中layers
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
        # # super()继承父类的构造函数
        super().__init__()
        self.num_heads = num_heads

        # 对每一个head的dim
        head_dim = dim // num_heads
        # 如果传入了qk_scale,就用传入的，
        # 否则使用每个头维度乘-0.5次方，对应attention公式中（QKT/根号d）*V的分母
        self.scale = qk_scale or head_dim ** -0.5
        # 定义q k v 的初始值
        # 使用一个全连接层直接得到qkv，全连接层节点个数*3=使用3个节点个数为dim的全连接层
        # 此处这样做的目的可能是为了并行化
        self.q = nn.Linear(dim, dim * 1, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 再定义一个全连接层，对应多头自注意输出b Concate拼接后，乘的W0 。输入输出节点个数都等于dim
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.fc1 = nn.Sequential(nn.Linear(256, 768), nn.ReLU())
        self.conv = nn.Conv1d(in_channels=256, out_channels=768, kernel_size=1, stride=1)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x, y):
        # x: (b, 768)
        # y: (b, 256)
        # x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        y = self.conv(y)
        y = rearrange(y, 'b c 1 -> b 1 c')
        Bx, Nx, Cx = x.shape     # (b, 1, 768)
        By, Ny, Cy = y.shape     # (b, 1, 768)

        q = self.q(x).reshape(Bx, Nx, 1, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)  # (b, 95, 1, 4, 196)
        kv = self.kv(y).reshape(By, Ny, 2, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)   # (b, 95, 2, 4, 196)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]

        q, k, v = q[0], kv[0], kv[1]  # (b, 4, 1, 196) # make torchscript happy (cannot use tensor as tuple)
        kv = (q @ k.transpose(-2, -1)) * self.scale

        kv = kv.softmax(dim=-1)
        kv = self.attn_drop(kv)
        kv = (kv @ v).transpose(1, 2).reshape(By, Nx, Cy)
        # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        kv = self.proj(kv)
        # 在进行一次dropout得到最终输出
        kv = self.proj_drop(kv)   # (b, 1, 768)
        # kv = rearrange(kv, '(b f) p c -> b f p c',  f=323)
        cro = x + kv

        return cro


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              # 设为16计算量更大
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


# if __name__ == '__main__':
#     input1 = torch.rand([2, 95, 3, 224, 224])
#     input2 = torch.rand([2, 95, 70, 3])
#     input3 = torch.rand([2, 95, 25, 3])
#     input4 = torch.rand([2, 95, 21, 3])
#     input5 = torch.rand([2, 95, 21, 3])
#     # # print(input1)
#     model = vit_base_patch16_224_in21k()
#     # print(model)
#     output1, loss1, loss2 = model(input1, input2, input3, input4, input5)
#     print(output1.shape)
