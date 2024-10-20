# views.py
import os
import tensorflow as tf
from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .serializers import ImageUploadSerializer
from PIL import Image

import numpy as np

class EfficientNetV2InferencerAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            
            img = Image.open(image)
            # Call the inferencer here
            result = self.run_inference(img)
            
            # Return the inference result
            return Response(result, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # def save_image(self, image):
    #     image_path = os.path.join(settings.MEDIA_ROOT, image.name)
    #     with open(image_path, 'wb+') as f:
    #         for chunk in image.chunks():
    #             f.write(chunk)
    #     return image_path


    def run_inference(self, image):        
        # Load the class labels
        classes = []
        with open('./models/label_map.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    classes.append(line)
        
        # Load model
        model_name = 'efficientnetv2-b0'
        model_dir = './models'
        image_size = 224  # Image size should match the model's expected input
        
        model = self.load_model(model_name, model_dir, len(classes))

        # Preprocess the image using PIL.Image
        image = image.resize((image_size, image_size))  # Resize to match model input size
        img_array = np.array(image) / 255.0  # Normalize the image to [0,1] range

        # Ensure the image has the right shape (batch size, height, width, channels)
        if img_array.shape[-1] == 4:
            # If the image has an alpha channel, discard it
            img_array = img_array[..., :3]
        
        img = np.expand_dims(img_array, 0)  # Add batch dimension

        # Run prediction
        logits = model(img, training=False)
        pred = tf.keras.layers.Softmax()(logits)
        top_prediction = tf.argsort(logits[0])[::-1][0].numpy()

        result = {
            'prediction': classes[top_prediction],
            'confidence': round(float(pred[0][top_prediction]), 4)
        }
        
        return result

    def load_model(self, model_name, model_dir, num_classes):
        fine_tuning = True
        trainable_layers_ratio = 0.4
        
        # Your FineTuningModel logic here (assumed to be in FineTuningModel.py)
        from .fine_tuning_service import FineTuningModel
        finetuning_model = FineTuningModel(model_name, None, False)
        model = finetuning_model.build(224, num_classes, fine_tuning, trainable_layers_ratio)
        
        best_model = os.path.join(model_dir, "best_model.h5")
        if not os.path.exists(best_model):
            raise Exception(f"Model file not found: {best_model}")
        
        model.load_weights(best_model)
        return model
