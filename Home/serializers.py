from rest_framework import serializers
from .models import Input, OutputImage

class input_serializer(serializers.Serializer):
    image = serializers.ImageField()

    def create(self, validated_data):
        return Input.objects.create(**validated_data)


class inputSerializer(serializers.ModelSerializer):
    inputImage=serializers.ImageField()
    class Meta:
        model=OutputImage
        fields=['inputImage','outputMask','outputImage']

class OutputSerializer(serializers.ModelSerializer):
    inputImage=serializers.ImageField()
    class Meta:
        model=OutputImage
        fields=['inputImage']