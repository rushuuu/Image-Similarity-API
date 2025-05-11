from fastapi import FastAPI, UploadFile, File, HTTPException
from enum import Enum
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import tempfile
import os
import logging
from typing import Dict, Tuple, List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Similarity API",
    description="API for comparing the similarity between two images using deep learning models",
    version="1.0.0"
)

# Enum for similarity categories
class SimilarityBucket(str, Enum):
    ALMOST_SAME = "Almost Same"
    VERY_SIMILAR = "Very Similar"
    SLIGHTLY_SIMILAR = "Slightly Similar"
    NOT_SIMILAR = "Not Similar"
    SAME = "Same"

# Device configuration (use GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Load models globally (once at startup)
def load_models() -> Dict[str, torch.nn.Module]:
    logger.info("Loading models...")
    models_dict = {
        "efficientnet_b0": models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT"),
        "vit_b_16": models.vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
    }

    for name, model in models_dict.items():
        logger.info(f"Processing model: {name}")
        if "vit" in name:
            model.heads = torch.nn.Identity()  # Remove classification head for ViT
        else:
            model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classification layer
        model.to(DEVICE).eval()  # Move to GPU (if available) and set to eval mode
        models_dict[name] = model
    
    logger.info("Models loaded successfully")
    return models_dict

# Initialize models
models_dict = load_models()

# Image preprocessing function
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess image for the model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image tensor ready for the model
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise ValueError(f"Error processing image: {str(e)}")

# Function to get similarity bucket based on score
def get_similarity_bucket(score: float) -> SimilarityBucket:
    """
    Convert a similarity score to a SimilarityBucket enum.
    
    Args:
        score: Similarity score between 0 and 1
        
    Returns:
        Appropriate SimilarityBucket enum value
    """
    if score > 0.95:
        return SimilarityBucket.SAME
    elif score >= 0.90:
        return SimilarityBucket.ALMOST_SAME
    elif score > 0.75:
        return SimilarityBucket.VERY_SIMILAR
    elif score > 0.60:
        return SimilarityBucket.SLIGHTLY_SIMILAR
    else:
        return SimilarityBucket.NOT_SIMILAR

# Compute similarity function
def compute_similarity(image1_path: str, image2_path: str) -> Tuple[float, SimilarityBucket]:
    """
    Compute similarity between two images using multiple models.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        
    Returns:
        A tuple containing (similarity_score, similarity_category)
    """
    similarities: List[float] = []

    for model_name, model in models_dict.items():
        logger.info(f"Computing features using {model_name}")
        img1 = preprocess_image(image1_path)
        img2 = preprocess_image(image2_path)

        with torch.no_grad():
            feat1 = model(img1).squeeze().flatten()
            feat2 = model(img2).squeeze().flatten()

        # Normalize feature vectors for better comparison
        feat1 = F.normalize(feat1, p=2, dim=0)
        feat2 = F.normalize(feat2, p=2, dim=0)
        
        similarity = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
        similarities.append(similarity)
        logger.info(f"{model_name} similarity: {similarity:.4f}")

    avg_similarity = sum(similarities) / len(similarities)
    logger.info(f"Average similarity score: {avg_similarity:.4f}")
    
    category = get_similarity_bucket(avg_similarity)
    logger.info(f"Similarity category: {category}")

    return avg_similarity, category

# FastAPI endpoints

@app.get("/")
def home():
    """Root endpoint that returns a welcome message."""
    return {
        "message": "Welcome to the Image Similarity API",
        "endpoints": {
            "/image_similarity": "POST endpoint for comparing two images",
            "/docs": "API documentation"
        }
    }

@app.get("/info")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"info": "rushil"}
            
@app.post("/image_similarity/")
async def image_similarity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two images and return their similarity score and category.
    
    Args:
        file1: First image file
        file2: Second image file
        
    Returns:
        JSON object with similarity_score and category
    """
    logger.info(f"Received request to compare images: {file1.filename} and {file2.filename}")
    
    # Validate file types
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    for file in [file1, file2]:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a supported image type. Supported types: {', '.join(allowed_extensions)}"
            )
    
    try:
        # Use temporary files to save uploaded images
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file1.filename)[1]) as temp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file2.filename)[1]) as temp2:

            temp1_path = temp1.name
            temp2_path = temp2.name
            
            # Save uploaded files to temporary files
            temp1.write(await file1.read())
            temp2.write(await file2.read())
            temp1.close()
            temp2.close()
            
            logger.info(f"Saved temporary files: {temp1_path} and {temp2_path}")

            # Compute similarity
            similarity_score, category = compute_similarity(temp1_path, temp2_path)

        # Cleanup temp files
        try:
            os.remove(temp1_path)
            os.remove(temp2_path)
            logger.info("Temporary files removed")
        except Exception as e:
            logger.warning(f"Error removing temporary files: {str(e)}")

        return {
            "similarity_score": round(similarity_score, 4),
            "category": category.value
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "models_loaded": list(models_dict.keys())}

# Not required in prod env
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
