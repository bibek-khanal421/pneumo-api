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
import os
import cv2
import sys
from io import BytesIO
from django.core.files.base import ContentFile
from segmentation_mask_overlay import overlay_masks
from django.core.files.uploadedfile import InMemoryUploadedFile

# a function to convert a pillow IMG file to file field django 
def convertToFileField(image):

    output = BytesIO()
    image.save(output, format='JPEG', quality=85)
    output.seek(0)
    new_pic= InMemoryUploadedFile(output, 'ImageField',
                                "temp.jpeg",
                                'image/jpeg',
                                sys.getsizeof(output), None)
    return new_pic

# create a overlaying of mask over the image
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_mask)
    img = img.resize((image.shape[1], image.shape[0]))
    img = np.array(img)
    img = cv2.addWeighted(image, 0.5, img, 0.5, 0)
    print(type(img))
    return img


@api_view(['GET', 'POST'])
def homeview(request):
    if request.method == 'POST':
        image = request.data['images']
        # obj=inputSerializer(image=image)
        # obj.save()
        image = Image.open(image)
        image = image.resize((256, 256), Image.ANTIALIAS).convert('RGB')
        model = get_model()
        model.load_weights(settings.WEIGHT_PATH)
        prediction = model.predict(np.array(image).reshape(1, 256, 256, 3)/255)
        predict_image = Image.fromarray(
            np.uint8(prediction.reshape(256, 256)*255)).convert('L')
        predict_image=overlay_mask(image, predict_image)
        predict_image.save('output.jpeg')
        # predict_image=np.dstack((np.array(predict_image), np.zeros((256,256,1)), np.zeros((256,256,1)).reshape(256,256,1))).reshape(256,256,3)
        # predict_image=np.concatenate((np.zeros((256,256)), np.zeros((256,256)), np.array(predict_image).reshape(256,256)), axis=0).reshape(256,256,3)
        # predict_image=Image.fromarray(np.uint8(predict_image))
        # plt.imshow(image,cmap='gray',alpha=1)
        # plt.imshow(predict_image, cmap='PRGn', alpha=0.25)
        # plt.savefig('o.png')

        final_image=overlay_masks( [np.array(predict_image).reshape(256,256,1)],np.array(image))
        final_image.savefig('o.png')
        # data=input_serializer(image=image)
        # data.save()
        return HttpResponse('done', content_type="text/plain")
        # response = {"msg": "Error processing the image!!!"}
        # json_data = JSONRenderer().render(response)
        # return HttpResponse(json_data, content_type="application/json")
    response = {"msg": "Error processing the image!!!"}
    json_data = JSONRenderer().render(response)
    return HttpResponse(json_data, content_type="application/json")


class MainView(APIView):
    def post(self, request, format=None):
        serializer=inputSerializer(data=request.data)
        if serializer.is_valid():
            imaged=serializer.validated_data['inputImage']
            image = Image.open(imaged)
            image = image.resize((256, 256), Image.ANTIALIAS).convert('RGB')
            model = get_model()
            model.load_weights(settings.WEIGHT_PATH)
            prediction = model.predict(np.array(image).reshape(1, 256, 256, 3)/255)
            predict_image = Image.fromarray(
            np.uint8(prediction.reshape(256, 256)*255)).convert('L')
            predict_image=np.dstack((np.array(predict_image), np.zeros((256,256,1)), np.zeros((256,256,1)).reshape(256,256,1))).reshape(256,256,3)
            # predict_image=np.concatenate((np.zeros((256,256)), np.zeros((256,256)), np.array(predict_image).reshape(256,256)), axis=0).reshape(256,256,3)
            predict_image=Image.fromarray(np.uint8(predict_image))
            predict_image=convertToFileField(predict_image)
            # plt.imshow(image,cmap='gray',alpha=1)
            # plt.imshow(predict_image, cmap='PRGn', alpha=0.25)
            # xy=Image.frombytes('RGB', 
            # plt.canvas.get_width_height(),plt.canvas.tostring_rgb())
            # img_io=StringIO()
            # predict_image.save(img_io, format='JPEG')
            # img_content=ContentFile(img_io.getvalue(),'img5.jpg')

            # final_image=overlay_masks( [np.array(predict_image).reshape(256,256,1)],np.array(image))
            # final_image.savefig('o.png')
            OutputImage(inputImage=imaged, outputMask=predict_image, outputImage=imaged).save()
            serializer.validated_data['outputImage']=predict_image
            serializer.validated_data['outputMask']=imaged
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            # serializer=OutputSerializer(outputMask=predict_image, outputImage=xy)
            # serializer.save()
            # return HttpResponse('done', content_type="text/plain")
            # return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request, format=None):
        images=OutputImage.objects.all()
        serializer=OutputSerializer(images, many=True)
        return Response(serializer.data)
        