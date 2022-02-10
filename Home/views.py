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
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from segmentation_mask_overlay import overlay_masks

# Create your views here.
def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5, 
    resize: tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    color = np.asarray(color).reshape(1,1,3)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined



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
