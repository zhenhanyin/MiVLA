from functools import partial
from turtle import update
import collections.abc as container_abcs
from itertools import repeat
import torch
import torch.nn as nn



def interpolate_pos_encoding(x, pos_embed, npatch, with_cls_token=True):
        # npatch = x.shape[1] - 1 if with_cls_token else x.shape[1]
        N = pos_embed.shape[1] - 1 if with_cls_token else pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        if with_cls_token:
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
        else:
            patch_pos_embed = pos_embed
        dim = x.shape[-1]
        #  h0 = h // self.patch_embed.patch_size
        #  w0 = w // self.patch_embed.patch_size
        #  w0, h0 = w0 + 0.1, h0 + 0.1
        L = npatch
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, N, 1, dim).permute(0, 3, 1, 2),
            size=(L, 1),
            # scale_factor=(L / N, 1),
            mode='bicubic',
        )
        assert L == patch_pos_embed.shape[2],"{} vs {}".format(L, patch_pos_embed.shape[2])
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if with_cls_token:
            patch_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return patch_pos_embed















def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qk = (q @ k.transpose(-2, -1)) * self.scale
        qk = qk.softmax(dim=-1)
        attn = self.attn_drop(qk)
        
        vv = ((v @ v.transpose(-2, -1)) * self.scale).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, [qk, vv]

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class AlignDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim=1024, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **args):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embed_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(embed_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        self.decoder_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.decoder_norm(self.last_layer(x))
        return x
    
class MaskedVisionTransformer(nn.Module):
    """ Masked Autoregressor with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, num_heads_in_last_block=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.rand(1, num_patches + 1, embed_dim), requires_grad=True) 

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(embed_dim, num_heads_in_last_block, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        # embed patches
        # h, w = x.size(2), x.size(3)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

class SAIPViT(MaskedVisionTransformer):
    """ Masked Autoregressor with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=(16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, num_heads_in_last_block=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, pretrained='',**args):
        super().__init__(img_size, patch_size, in_chans,
                 embed_dim, depth, num_heads, num_heads_in_last_block,
                 mlp_ratio, norm_layer)

        # --------------------------------------------------------------------------
        self.out_index = [6]
        self.inter_norm = norm_layer(embed_dim)
        self.anchor_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.anchor_patches = self.anchor_size[0] * self.anchor_size[1]
        self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained=''):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        if pretrained != '':
            self.init_from_pretrain(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_from_pretrain(self, pretrained=None):
        checkpoint_model = torch.load(pretrained, map_location='cpu')
        if 'model' in checkpoint_model:
            param_dict = checkpoint_model['model']
        elif 'state_dict' in checkpoint_model:
            param_dict = checkpoint_model['state_dict']
        elif 'student' in checkpoint_model: ### for dino
            param_dict = checkpoint_model["student"]
        else:
            param_dict = checkpoint_model
        param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
        count=0
        for k, v in param_dict.items():
            if k not in self.state_dict().keys():
                continue
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                print('shape resize from :{}: param_dict{} to self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
                b, l, d = self.state_dict()[k].size()
                pos_emb = v[:,1:,:]
                pos_emb = torch.nn.functional.interpolate(pos_emb.unsqueeze(0), size=(l-1,d),mode='bilinear')[0]
                v = torch.cat([v[:,0,:].unsqueeze(1), pos_emb], dim=1)
                param_dict[k] = v
            try:
                self.state_dict()[k].copy_(v)
                count +=1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))
        msg = self.load_state_dict(param_dict, strict=False)
        print(msg)
        
    def get_mask(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        L = self.anchor_patches
        N = x.shape[0]
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1.
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask
    
    def random_masking_with_ref_anchor(self, masked_x, mask_ratio, hw_size):
        
        B, L, D = masked_x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        base_mask = self.get_mask(masked_x, mask_ratio)
        base_mask = base_mask.view(B, 1, self.anchor_size[0], self.anchor_size[1])
        mask = torch.nn.functional.interpolate(base_mask, hw_size)
        mask = mask.flatten(1)
        ids_shuffle = torch.argsort(1-mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(masked_x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, [base_mask, 1-mask], ids_restore
    
    def forward(self, x, mask_ratio, with_anchor_mask=False, get_intermediate_only=False):
        # embed patches
        
        ph, pw = x.size(2)//self.patch_size[0], x.size(3)//self.patch_size[1]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = interpolate_pos_encoding(x, self.pos_embed, x.shape[1])
        x = x + pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0.:
            if with_anchor_mask:
                x, mask, ids_restore = self.random_masking_with_ref_anchor(x, mask_ratio, (ph, pw))
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        _outs = []
        
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            if i in self.out_index:
                _outs.append(self.inter_norm(x))
                if get_intermediate_only:
                    return _outs, mask, ids_restore, attn
            
        _outs.append(self.norm(x))
        
        return _outs, mask, ids_restore, attn


class SAIPKDViTWrapper(nn.Module):
   
    def __init__(self, **args):
        super(SAIPKDViTWrapper, self).__init__()
        self.backbone = SAIPViT(**args)
        self.adapter = AlignDecoder(**args) 

    def from_pretrained(self, pretrained=None):
        ckpt = torch.load(pretrained, map_location='cpu')
        if 'student' in ckpt.keys():
            ckpt = ckpt['student']
        
        ckpt = {k.replace('module.',''): v for k, v in ckpt.items()}
        missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def forward(self, base_imgs):
        base_inputs = base_imgs
        outputs = {}
        b_feats, _, _, aligned_atten = self.backbone(base_inputs, 0.)
        b_aligned_feats = b_feats[1]
        b_aligned_feats = self.adapter(b_aligned_feats)
        aligned_cls_feats = [b_aligned_feats[:, 0, :]]
        aligned_patch_feats = b_aligned_feats[:, 1:, :]
        # img_level_aligned_tokens = torch.cat(aligned_cls_feats)
        # outputs['aligned_cls_feats'] = img_level_aligned_tokens
        # outputs['aligned_patch_feats'] = aligned_patch_feats
        # outputs['qkv_atten'] =  aligned_atten
        return aligned_patch_feats

    

def saipv1_kd_vit_tiny_patch14(pretrained=None, **kwargs): 
    print('DPAL-ViT/Ti-14')
    model = SAIPKDViTWrapper(patch_size=(14, 14), embed_dim=192, depth=12, 
        num_heads=6, num_heads_in_last_block=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        model.from_pretrained(pretrained)
    model.embed_dim = 1024
    return model


if __name__ == '__main__':
    model = saipv1_kd_vit_tiny_patch14(pretrained='/home/yinzhenhan/Human_Pre/H_RDT/bak/dino-siglip/checkpoint.pth', img_size=(224, 224))
    dumpy_input = torch.rand(2,3,384,384)
    model(dumpy_input)