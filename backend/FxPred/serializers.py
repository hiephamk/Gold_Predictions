from rest_framework import serializers
from .models import *

class PredictedDataBaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedDataBase
        fields = '__all__'
class PredictedData_15m_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedData_15m
        fields = '__all__'
class PredictedData_30m_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedData_30m
        fields = '__all__'
class PredictedData_1h_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedData_1h
        fields = '__all__'
class PredictedData_4h_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedData_4h
        fields = '__all__'
class PredictedData_1d_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedData_1d
        fields = '__all__'
class PredictedData_1wk_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedData_1wk
        fields = '__all__'
