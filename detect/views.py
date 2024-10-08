from django.http import JsonResponse, FileResponse
from rest_framework.decorators import api_view
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TFSMLayer
from django.conf import settings
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
from io import BytesIO
from .gemini import asker
@api_view(['GET'])
def history(request):
    # Retrieve session history
    session_data = request.session.get('history', [])
    return JsonResponse(session_data, safe=False)


model = load_model('./models/model.keras')

# Image processing helper functions
def load_and_preprocess_image(image_file, img_dim=(1536, 128)):
    image = Image.open(image_file).convert('RGB')
    image = image.resize(img_dim)
    img_array = np.array(image)
    img_array = img_array[:, :, ::-1]  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    return img_array, image

from PIL import Image,ImageOps, UnidentifiedImageError
from io import BytesIO
from django.http import JsonResponse, FileResponse
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# Assuming you have the function for drawing a red circle
def draw_red_circle(image, center, radius=50):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="red", width=5)
    return image



@api_view(['POST'])
def classify_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        try:
            img = Image.open(image_file)
            img = img.resize((1536, 128))  

            img_array = np.array(img)
            if img_array.shape[:2] != (128, 1536):
                return JsonResponse({
                        'error': 'Invalid image dimensions. Please upload a biopsy slide image with dimensions 1536x128.'
                    }, status=400)
            # If the image dimensions are (128, 1536, 3), transpose them to (1536, 128, 3)
            if img_array.shape[:2] == (128, 1536):
                img_array = np.transpose(img_array, (1, 0, 2))

            # Add batch dimension and preprocess the image for VGG16
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Use the original PIL image object (img) as the original image
            original_image = img

            # Perform prediction
            prediction = model.predict(img_array)

            predicted_class = np.argmax(prediction, axis=1)[0]
            score = prediction[0][predicted_class]

            # Map the predicted class (ISUP grade) to Gleason score
            gleason_score_map = {
                0: "Gleason Score 6",
                1: "Gleason Score 7 (3 + 4)",
                2: "Gleason Score 7 (4 + 3)",
                3: "Gleason Score 8",
                4: "Gleason Score 9",
                5: "Gleason Score 10",
            }
            
            gleason_score = gleason_score_map.get(predicted_class, "Unknown Gleason Score")

            # Assuming prediction gives some coordinates for the area to highlight
            image_center = (original_image.width // 2, original_image.height // 2)
            
            # Draw a red circle around the prediction area
            processed_image = draw_red_circle(original_image, image_center)

            # Save the processed image to a buffer
            buffer = BytesIO()
            processed_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Return the processed image and the classification result
            # response = FileResponse(buffer, as_attachment=True, filename="processed_image.png")
            image_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.png')
            processed_image.save(image_path, format="PNG")

            # Return the image URL (assuming MEDIA_URL is configured)
            image_url = 'processed_image.png'
            full = asker(gleason_score)
            return JsonResponse({
            'predicted_class': int(predicted_class),
            'score': float(score),
            'gleason_score': gleason_score,  # Return Gleason score
            'image_url':image_url,
            'full': full
        })
        
        

        except ValueError as e:
            return JsonResponse({
                'error': f'{e}Image processing failed. Please ensure the image is a valid biopsy slide.'
            }, status=400)

        except Exception as e:
            # Catch any other unforeseen errors
            return JsonResponse({
                'error': f'An unexpected error occurred: {str(e)}'
            }, status=500)


