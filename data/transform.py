# data/transform.py
from torchvision import transforms

class MultiVisualTransform:
    def __init__(self, size=224):
        # 1. 基础变换 (Resize, ToTensor)
        self.base_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        
        # 2. CLIP 专用归一化
        self.norm_clip = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # 3. BioMedCLIP / DINOv2 通用归一化 (ImageNet)
        self.norm_imagenet = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )

    def __call__(self, image):
        # image: PIL Image
        base_tensor = self.base_transform(image)
        
        # 返回字典，供 Dataset 封装
        return {
            "clip_img": self.norm_clip(base_tensor.clone()),
            "med_img": self.norm_imagenet(base_tensor.clone()), # BioMedCLIP
            "dino_img": self.norm_imagenet(base_tensor.clone()) # DINOv2
        }