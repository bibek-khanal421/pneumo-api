from PIL import Image
import cv2
import numpy as np
from tensorflow import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
import os
current_directory = os.getcwd()
# defining the model path
model_path_0 = os.path.join(current_directory,"Home\ml_models\Adam-Dice\segmentation.h5")
# model_path_1 = os.path.join(current_directory,"Home\ml_models\dice_90\segmentation.h5")

# defining the parameters
width = 256
height = 256
image_channel = 3
mask_channel = 1
BACKBONE = 'seresnext50'
top_thresh = 0.5
bot_thresh = 0.2
min_area = 200


def api_handler(image):
    # resizing the images
    image = image.reshape(1, width, height, image_channel)
    # defining and compiling the model
    def get_model(model_path):
        keras.backend.clear_session()
        model = sm.Unet(backbone_name=BACKBONE, input_shape=(
            width, height, image_channel), encoder_weights='imagenet', encoder_freeze=True)
        model.load_weights(model_path)
        return model

    # image post processing
    def post_process_image(data, top_score_threshold, bot_score_threshold, min_contour_area):
        temp = []
        for predicted in data:
            predicted = np.array(predicted)
            classification_mask = predicted > top_score_threshold
            mask = predicted.copy()
            print(classification_mask.sum(axis=(0, 1, 2)))
            mask[classification_mask.sum(
                axis=(0, 1, 2)) < min_contour_area, :, :, :] = np.zeros_like(predicted[0])
            mask = mask > bot_score_threshold
            temp.append(mask.astype(np.uint8))
        return np.array(temp).reshape(width, height, mask_channel)

    def overlay(image, mask):
        mask = np.array(mask).reshape(width, height)
        image = np.array(image).reshape(width, height, image_channel)
        mask_bool = mask == 1
        image[:, :, 0][mask_bool] = 255
        image[:, :, 1][mask_bool] = 0
        image[:, :, 2][mask_bool] = 0
        return image.reshape(width, height, image_channel)

    model = get_model(model_path_0)
    # initializing the prediction
    prediction_0 = np.array(model.predict(
        image/255)).reshape(width, height, mask_channel)
    # model = get_model(model_path_1)
    # initializing the prediction
    # prediction_1 = np.array(model.predict(
    #     image/255)).reshape(width, height, mask_channel)
    tmp = np.stack((prediction_0, prediction_0), axis=2)
    tmp = np.array(tmp).reshape(width, height, 2)
    # combining the predictions
    prediction = np.mean(tmp, axis=2).reshape(1, width, height, mask_channel)
    prediction = post_process_image(prediction, top_score_threshold=top_thresh,
                                    bot_score_threshold=bot_thresh, min_contour_area=min_area)
    prediction_temp=prediction.copy()
    prediction = overlay(image[0], prediction)
    return prediction, np.array(prediction_temp*255).reshape(width, height)


# if __name__ == "__main__":
#     image = Image.open(r"\pneumothorax\0_test_1_.png")
#     image = image.resize((256, 256), Image.ANTIALIAS).convert('RGB')
#     prediction = api_handler(np.array(image))
#     cv2.imshow("prediction", prediction)
#     cv2.waitKey(0)