
from rest_framework import serializers
from apps.models import Inputargs
class Input_Serializer(serializers.ModelSerializer):
    class Meta:
        model =    Inputargs
        read_only_fields = ("id_args","args", "created_at")
        fields = read_only_fields
        
