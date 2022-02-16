from django.db import models

# Create your models here.
class Input(models.Model):
    image = models.ImageField()


class OutputImage(models.Model):
    inputImage=models.ImageField()
    outputMask=models.ImageField(blank=True, null=True)
    outputImage=models.ImageField(blank=True, null=True)