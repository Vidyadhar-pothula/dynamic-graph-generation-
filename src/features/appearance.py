import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class AppearanceExtractor:
    def __init__(self, device="mps"):
        self.device = device
        print(f"Initializing ResNet18 visual extractor on device: {self.device}")
        
        # Load pre-trained ResNet18
        # Use modern weights interface
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            # Fallback for older torchvision
            self.model = models.resnet18(pretrained=True)
            
        # Strip final classification layer to get 512-dimensional output
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms for ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    @torch.no_grad()
    def extract_crop_feature(self, frame_cv, bbox):
        """
        Crops the bounding box from frame_cv, preprocesses it, and passes it through ResNet18.
        frame_cv: OpenCV image (BGR)
        bbox: [xmin, ymin, xmax, ymax]
        Returns:
            numpy array of shape (512,)
        """
        xmin, ymin, xmax, ymax = bbox
        h, w, _ = frame_cv.shape
        
        # Clamp coordinates
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))
        
        # Crop image and convert to RGB PIL Image
        crop = frame_cv[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            # Empty crop fallback
            return torch.zeros(512).numpy()
            
        # Convert BGR to RGB
        crop_rgb = cv2_to_pil = Image.fromarray(crop[:, :, ::-1])
        
        # Apply transformations
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        # Forward pass
        feature = self.model(tensor).squeeze(0).cpu().numpy()
        return feature
        
    @torch.no_grad()
    def extract_batch_crops(self, frame_cv, bboxes):
        """
        Extracts features for a list of bounding boxes in batch.
        """
        if not bboxes:
            return []
            
        h, w, _ = frame_cv.shape
        tensors = []
        
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(0, min(xmax, w - 1))
            ymax = max(0, min(ymax, h - 1))
            
            crop = frame_cv[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                tensors.append(torch.zeros(3, 224, 224))
                continue
                
            crop_rgb = Image.fromarray(crop[:, :, ::-1])
            tensors.append(self.transform(crop_rgb))
            
        # Stack all crops into a single batch tensor
        batch_tensor = torch.stack(tensors).to(self.device)
        features = self.model(batch_tensor).cpu().numpy()
        
        return [f for f in features]
