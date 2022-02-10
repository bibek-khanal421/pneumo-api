from django.shortcuts import render
from rest_framework.renderers import JSONRenderer
from .serializers import input_serializer
from django.http import HttpResponse
from .models import Input
from rest_framework.decorators import api_view
# from rest_framework.response import Response
from django.conf import settings
from .MLmodel import get_model
from PIL import Image
import numpy as np
import os
import cv2

# Create your views here.


@api_view(['GET', 'POST'])
def homeview(request):
    if request.method == 'POST':
        image = request.data['images']
        # base = settings.BASE_DIR
        # image = Image.open(os.path.join(base,obj.image.url))
        image = Image.open(image)
        image = image.resize((256, 256), Image.ANTIALIAS).convert('RGB')
        model = get_model()
        model.load_weights(settings.WEIGHT_PATH)
        prediction = model.predict(np.array(image).reshape(1, 256, 256, 3)/255)
        predict_image = Image.fromarray(
            np.uint8(prediction.reshape(256, 256)*255)).convert('L')
        predict_image = predict_image.convert('RGB')

        # print(np.array(predict_image).shape)
        # image.paste(image, (0, 0), predict_image)
        predict_image.save('o.png')

        return HttpResponse('done', content_type="text/plain")
        # response = {"msg": "Error processing the image!!!"}
        # json_data = JSONRenderer().render(response)
        # return HttpResponse(json_data, content_type="application/json")
    response = {"msg": "Error processing the image!!!"}
    json_data = JSONRenderer().render(response)
    return HttpResponse(json_data, content_type="application/json")


# @api_view(['GET', 'POST'])
# def homeview(request):
#     if request.method == 'POST':
#         Input.objects.all().delete()
#         serializer = input_serializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             obj = Input.objects.get()
#             # base = settings.BASE_DIR
#             image = Image.open(obj.image)
#             image = image.resize((256,256), Image.ANTIALIAS).convert('RGB')
#             model = get_model()
#             model.load_weights(settings.WEIGHT_PATH)
#             prediction = model.predict(np.array(image).reshape(1,256,256,3)/255)
#             # predict_image = Image.fromarray(np.array(prediction).reshape(256,256))
#             predict_image = Image.fromarray(np.uint8(prediction.reshape(256,256)*255)).convert('L')
#             image.paste(Image.open(obj.image).resize((256,256), Image.ANTIALIAS).convert('L'), (0, 0), predict_image)
#             image.save('o.png')
#             # print(predict_image.shape)
#             obj.image.delete()
#             obj.delete()
#             return HttpResponse(Image.open('o.png'), content_type="image/png")
#             # response = {"msg": "Error processing the image!!!"}
#             # json_data = JSONRenderer().render(response)
#             # return HttpResponse(json_data, content_type="application/json")
#         response = {"msg": "Error processing the image!!!"}
#         json_data = JSONRenderer().render(response)
#         return HttpResponse(json_data, content_type="application/json")
