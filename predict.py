# predict.py
import torch
from torchvision import transforms
from PIL import Image
from model import create_resnet50
import numpy as np

def predict(img: Image.Image, model_path='best_model.pth'):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        return {"error": f"Model file not found at {model_path}. Ensure it is in the same directory as the server."}
    
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise KeyError("Checkpoint missing 'classes' key. Please ensure your 'best_model.pth' is correctly saved with class names.")


    model = create_resnet50(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    img = img.convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)

 
    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)

        top_p, top_class = probs.topk(3, dim=1)
        
 
        top_probs = top_p.cpu().numpy().flatten().tolist()
        top_classes_idx = top_class.cpu().numpy().flatten().tolist()
        
        top_n_predictions = []
        for i in range(len(top_probs)):
            top_n_predictions.append({
                'breed': classes[top_classes_idx[i]],
                'score': top_probs[i]
            })

    main_prediction = top_n_predictions[0]

    return {
        'prediction': main_prediction['breed'],
        'confidence': main_prediction['score'],
        'top_n_predictions': top_n_predictions
    }

if __name__ == '__main__':
    print("This file is primarily for API usage.")
