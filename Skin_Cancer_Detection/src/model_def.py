import torch
import torch.nn as nn
import timm

class HybridSkinModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(HybridSkinModel, self).__init__()
        

        self.cnn = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)
        
      
        self.bridge = nn.Conv2d(320, 192, kernel_size=1)
        
      
        vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
        self.transformer_blocks = vit_model.blocks
        self.norm = vit_model.norm
        
      
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
       
        features = self.cnn(x)[-1] 
        
      
        x = self.bridge(features)    
        
       
        x = x.flatten(2).transpose(1, 2) 
        

        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        
        x = x.mean(dim=1) 
        logits = self.classifier(x)
        
        return logits


if __name__ == "__main__":
    model = HybridSkinModel(num_classes=7)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model initialized. Output shape: {output.shape}")