from rest_framework import serializers
from .models import Input

class input_serializer(serializers.Serializer):
    image = serializers.ImageField()

    def create(self, validated_data):
        return Input.objects.create(**validated_data)