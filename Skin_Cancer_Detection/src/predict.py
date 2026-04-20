import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import torch.nn.functional as F

# 1. ADD THIS: This allows predict.py to see model_def.py in the same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. UPDATED IMPORT: Use a direct import instead of a relative one
try:
    from model_def import HybridSkinModel
except ImportError:
    from .model_def import HybridSkinModel

class Predictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "model", "hybrid_model.pth")
        
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model not found at {model_path}")
            
        self.model = HybridSkinModel(num_classes=7).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded successfully from {model_path}")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Standard HAM10000 Alphabetical Order
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_names = {
            'akiec': 'Actinic keratoses',
            'bcc': 'Basal cell carcinoma',
            'bkl': 'Benign keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic nevi',
            'vasc': 'Vascular lesions'
        }

    def run(self, image_path_or_pil):
        """Returns the full probability tensor for all classes."""
        if isinstance(image_path_or_pil, str):
            if not os.path.exists(image_path_or_pil):
                # Return a tensor of zeros if file not found to prevent crashes
                return torch.zeros(len(self.classes))
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
            
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            # We return the raw probabilities so app.py can use them for the chart
            probs = F.softmax(output, dim=1)[0]
            
        return probs # Returns TENSOR, not tuple!

# --- TEST BLOCK ---
if __name__ == "__main__":
    predictor = Predictor()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Change this filename to a real one in your folder
    test_image_filename = "ISIC_0024306.jpg" 
    test_image_path = os.path.join(project_root, "Data", "all_images", test_image_filename)
    
    if os.path.exists(test_image_path):
        print(f"🔍 Analyzing image: {test_image_filename}...")
        
        # In the test block, we handle the output manually
        probs = predictor.run(test_image_path)
        conf, idx = torch.max(probs, dim=0)
        
        disease_type = predictor.class_names[predictor.classes[idx.item()]]
        
        print("-" * 30)
        print(f"🩺 Prediction: {disease_type}")
        print(f"📊 Confidence: {conf.item()*100:.2f}%")
        print("-" * 30)
    else:
        print(f"❌ Error: Image not found at {test_image_path}")