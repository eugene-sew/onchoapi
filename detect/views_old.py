from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import os
# from .model_inference import load_model, predict, save_mask_as_image,create_zip_with_predictions
import nibabel as nib
import zipfile
import numpy as np  
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .model_inference import load_model, predict, save_mask_as_image_with_circles, create_zip_with_predictions

model = load_model(os.path.join("./models/udmodel.pth"))
# model = load_model(os.path.join("./models/udmodel.pth"))

detect_request_body = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'file': openapi.Schema(type=openapi.TYPE_FILE, description='NIfTI file to process'),
    },
)


detect_response_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'result_nii': openapi.Schema(type=openapi.TYPE_STRING, description='Processed NIfTI file name'),
        'result_png': openapi.Schema(type=openapi.TYPE_STRING, description='Generated PNG file name'),
    },
)


@swagger_auto_schema(method='get', responses={200: openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING))})
@api_view(['GET'])
def history(request):
    # Retrieve session history
    session_data = request.session.get('history', [])
    return JsonResponse(session_data, safe=False)



# Define the request and response schema for Swagger documentation
detect_request_body = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'file': openapi.Schema(type=openapi.TYPE_FILE, description='NIfTI file to process'),
    },
)

detect_response_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'result_nii': openapi.Schema(type=openapi.TYPE_STRING, description='Processed NIfTI file name'),
        'result_png': openapi.Schema(type=openapi.TYPE_STRING, description='Generated PNG file name'),
    },
)

# Retrieve session history (if required)
@swagger_auto_schema(method='get', responses={200: openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING))})
@api_view(['GET'])
def history(request):
    session_data = request.session.get('history', [])
    return JsonResponse(session_data, safe=False)

# Handle NIfTI file upload, segmentation, and response
@swagger_auto_schema(method='post', request_body=detect_request_body, responses={200: detect_response_schema, 400: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)})})
@api_view(['POST'])
def detect(request):
    if 'nii_file' not in request.FILES:
        return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

    uploaded_file = request.FILES['nii_file']
    result_nii = uploaded_file.name

    # Save the uploaded NIfTI file
    temp_path = default_storage.save('temp.nii', ContentFile(uploaded_file.read()))

    # Get model prediction (mask)
    predicted_mask = predict(model, temp_path)

    # Save the predicted mask as a NIfTI file
    output_nii = nib.Nifti1Image(predicted_mask, np.eye(4))
    output_nii_path = default_storage.save('prediction_output.nii', ContentFile(b''))
    nib.save(output_nii, default_storage.path(output_nii_path))

    # Define template for output image slices
    output_image_path_template = 'slice_{}.png'

    # Save mask slices as images with cancer regions circled in red
    slice_image_paths = save_mask_as_image_with_circles(predicted_mask, output_image_path_template)

    # Create ZIP file with NIfTI and images
    zip_filename = create_zip_with_predictions(output_nii_path, slice_image_paths)

    # Return the zip file in the response
    return JsonResponse({"result_nii": result_nii, 'download': zip_filename})
