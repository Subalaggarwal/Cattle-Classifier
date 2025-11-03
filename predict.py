# predict.py
import torch
from torchvision import transforms
from PIL import Image
from model import create_resnet50
import numpy as np

def predict(img: Image.Image, model_path='best_model.pth'):
    """
    Predicts the breed of cattle from a PIL Image object.
    
    Args:
        img (PIL.Image.Image): The input image to analyze (This is the critical change).
        model_path (str): Path to the saved PyTorch model checkpoint.
        
    Returns:
        dict: A dictionary containing the top prediction and top N predictions, or an error dictionary.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint safely
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        return {"error": f"Model file not found at {model_path}. Ensure it is in the same directory as the server."}
    
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise KeyError("Checkpoint missing 'classes' key. Please ensure your 'best_model.pth' is correctly saved with class names.")

    # Initialize model (same structure as training)
    # Assumes create_resnet50 is imported from model.py
    model = create_resnet50(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    # Preprocessing same as training (inference transforms from dataset.py)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Preprocess image
    # The 'img' variable here is already the PIL Image object passed by app_server.py
    img = img.convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)
        
        # Get top 3 predictions
        top_p, top_class = probs.topk(3, dim=1)
        
        # Convert to lists for JSON serialization
        top_probs = top_p.cpu().numpy().flatten().tolist()
        top_classes_idx = top_class.cpu().numpy().flatten().tolist()
        
        top_n_predictions = []
        for i in range(len(top_probs)):
            top_n_predictions.append({
                'breed': classes[top_classes_idx[i]],
                'score': top_probs[i]
            })

    # The main prediction is the first one in the list
    main_prediction = top_n_predictions[0]

    return {
        'prediction': main_prediction['breed'],
        'confidence': main_prediction['score'],
        'top_n_predictions': top_n_predictions
    }

if __name__ == '__main__':
    # Local Test Block
    print("This file is primarily for API usage.")
