from typing import Callable, Optional, Tuple, Union
import torch.nn as nn

def get_tuple(n):
    return (n, n)

class PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer: Optional[Callable] = None,
            flatten=True,
            bias=True,
            strict_img_size = True
    ):
        super().__init__()
        self.patch_size = get_tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        ## We will want to flatten the image to treat it 
        ## like sequential text embeddings for learning 
        ## in the attention network
        self.flatten = flatten # True

        self.strict_img_size = strict_img_size # True

        ## in_chans -> channel: 3
        ## embed_dim -> 1152
        ## kernel_size -> 2
        ## stride -> 2
        ## bias -> true
        ## Input should be images in the following format (N, C, H, W)
        ## or (C, H, W). The output from this convolution will be 
        ## (N, embed_dim, H // 2, W // 2) or (embed_dim, H // 2, W // 2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

        ## We just use a placeholder for a nn layer
        ## whatever tensor fed in will get itself back.
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        '''
        With the patch size that we want specified, we are finding 
        the largest number of grid the that image size can be 
        broken down into corresponding to the patch size (both in the 
        x and y direction). From there, we can find the total number of
        output patches that will break the image up nicely.
        If image size is (32, 32), patch size is 5, our output
        will be ((32,32), (6,6), 36). This is because we have a 6,6 grid
        formed with 6 * 6 number of patches of the original image.
        '''
        if img_size is None:
            return None, None, None
        img_size = get_tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches


    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0]
                assert W == self.img_size[1]

        ## Apply convolution to the current image
        x = self.proj(x)
        if self.flatten:
            
            ## We start flattening from index 2 of dimensions
            ## of the image. (15, 4, 32, 32) -> (15, 4, 1024)
            ## We then swap index 1 with index 2 of the image 
            ## dimension. -> (15, 1024, 4)
            x = x.flatten(2).transpose(1, 2) 

        x = self.norm(x)
        return x
    