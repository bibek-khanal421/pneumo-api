from django.shortcuts import render
from rest_framework.renderers import JSONRenderer
from .serializers import input_serializer, inputSerializer, OutputSerializer
from django.http import HttpResponse
from .models import Input, OutputImage
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from django.conf import settings
from .MLmodel import get_model
from PIL import Image
from rest_framework import status
from rest_framework.response import Response
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import cv2
from .api_handler import api_handler
import sys
from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile

# a function to convert a pillow IMG file to file field django


def convertToFileField(image):

    output = BytesIO()
    image.save(output, format='JPEG', quality=85)
    output.seek(0)
    new_pic = InMemoryUploadedFile(output, 'ImageField',
                                   "temp.jpeg",
                                   'image/jpeg',
                                   sys.getsizeof(output), None)
    return new_pic


class MainView(APIView):
    def post(self, request, format=None):
        serializer = inputSerializer(data=request.data)
        if serializer.is_valid():
            imaged = serializer.validated_data['inputImage']
            image = Image.open(imaged)
            image = np.array(image.resize(
                (256, 256), Image.ANTIALIAS).convert('RGB'))
            prediction, prediction_mask = api_handler(image)
            prediction_mask = Image.fromarray(prediction_mask)
            prediction = Image.fromarray(prediction)
            prediction.save("o.png")
            prediction = convertToFileField(prediction)
            prediction_mask = convertToFileField(prediction_mask)
            OutputImage(inputImage=imaged, outputMask=prediction_mask,
                        outputImage=prediction).save()
            serializer.validated_data['outputImage'] = prediction
            serializer.validated_data['outputMask'] = prediction_mask
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, format=None):
        images = OutputImage.objects.all()
        serializer = OutputSerializer(images, many=True)
        return Response(serializer.data)
