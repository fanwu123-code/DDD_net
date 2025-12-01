import torch
from torch import nn
import torchvision
from torchvision.transforms import Resize
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINOv2_Encoder(nn.Module):
    def __init__(self, model_name='dinov2_vitl14', encoded_image_size=14):
        super(DINOv2_Encoder, self).__init__()
        self.net = torch.hub.load('facebookresearch/dinov2', model_name)
        self.enc_image_size = encoded_image_size
        if 'vitl' in model_name:
            hidden_dim = 1024
            self.projection1 = nn.Conv2d(hidden_dim, 512, kernel_size=1)
            self.projection2 = nn.Conv2d(hidden_dim, 512, kernel_size=1)
            self.projection3 = nn.Conv2d(hidden_dim, 512, kernel_size=1)
        elif 'vitb' in model_name:
            hidden_dim = 768
            self.projection1 = nn.Conv2d(hidden_dim, 512, kernel_size=1)
            self.projection2 = nn.Conv2d(hidden_dim, 512, kernel_size=1)
            self.projection3 = nn.Conv2d(hidden_dim, 512, kernel_size=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        
        self.fine_tune()

    def forward(self, images):
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            torch_resize = Resize([224, 224])
            images = torch_resize(images)
        
        batch_size = images.shape[0]
        
        block_indices = [3, 11, 23]
        
        intermediate_outputs = self.net.get_intermediate_layers(
            images, 
            n=block_indices,
            return_class_token=False
        )
        
        features = []
        h = w = int(images.shape[2] // self.net.patch_size)
        for output in intermediate_outputs:
            feat = output.permute(0, 2, 1).reshape(batch_size, -1, h, w)
            features.append(feat)
        
        out_large = self.projection1(features[0])
        out_media = self.projection2(features[1])
        out_small = self.projection3(features[2])
        
        out_large = self.adaptive_pool(out_large)
        out_media = self.adaptive_pool(out_media)
        out_small = self.adaptive_pool(out_small)
        
        return out_large, out_media, out_small

    def fine_tune(self, fine_tune=False):
        for p in self.net.parameters():
            p.requires_grad = False
        
        if fine_tune:
            for i in range(len(self.net.blocks) - 1, len(self.net.blocks)):
                for p in self.net.blocks[i].parameters():
                    p.requires_grad = True








