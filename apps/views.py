from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import subprocess
import json
import shutil
import argparse
import os
import sys
import pandas as pd
import subprocess
import tempfile
from os.path import splitext
from apps.models import Inputargs
from apps.serializers import Input_Serializer
from celery import shared_task
from .tasks import got_to_s
from apps.Cloned2.lib.PBS.arguments_Maker import args
from apps.Cloned2.lib.PBS.pyspark_argument_Maker import argss
from apps.Cloned2.bin.DSEMainPipeline import Decision_M
#from apps.Cloned2.bin.PySparkMainPipeline import Decision_M_Pyspark


@shared_task()
def adding_task(a, b):
    return a + b


# @shared_task
# def Run_dse_pipleline(pd_df,df):    
#     dname,target,sklearn,algorithm,config_object_user,ai ,model_file,predict,phenome_data,user_defined_terminology,sample_type,description,uom_type,all_M,d_col,index,to_csv_name,cut_off,output=args(pd_df,df)
#     saveout = sys.stdout
#     fsock = open(output, 'w')
#     sys.stdout = fsock
#     Decision_M(dname,target,sklearn,algorithm,config_object_user,ai ,model_file,predict,phenome_data,
#         user_defined_terminology,sample_type,description,uom_type,all_M,d_col,index,to_csv_name,cut_off
#             )
#     sys.stdout = saveout
#     fsock.close()
#     f = open(output, "r")
#     df = f.read()
#     return df
class Dse_pipeline(APIView):
    def get(self):
        task = adding_task.delay(2, 5)
        print(f"id={task.id}, state={task.state}, status={task.status}")
        args = Inputargs.objects.all()
        serializer = Input_Serializer(args, many=True)
        return Response(f"id={task.id}, state={task.state}, status={task.status}")  # serializer.data)

    def post(self, data):
        #         try:
        df = json.loads(data.body)
        json_args = Inputargs.objects.create(args=df)
        json_args.save()
        pd_df = pd.DataFrame.from_dict(df)
        dname, target, sklearn, algorithm, config_object_user, ai, model_file, predict, phenome_data, user_defined_terminology, sample_type, description, uom_type, all_M, d_col, index, to_csv_name, min_depth, max_depth, min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors, xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_C, svm_gamma, svm_kernel, default, cut_off, output, analysis_id, db_name, N_features, pca, learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping, dataset_type, test_size, samples, filter_M ,vifNo, VIF, Outliers= args(
            pd_df, df)
        print(pca())
        #             saveout = sys.stdout
        #             fsock = open(output, 'w')
        #             sys.stdout = fsock
        got_to_s(dname, target, sklearn, algorithm, config_object_user, ai, model_file, predict, phenome_data,
                 user_defined_terminology, sample_type, description, uom_type, all_M, d_col, index, to_csv_name,
                 min_depth, max_depth, min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors,
                 xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_C, svm_gamma, svm_kernel,
                 default, analysis_id, db_name, N_features,filter_M, pca, learning_rate_init, max_iter, hidden_layer_sizes,
                 activation, alpha, early_stopping, dataset_type, test_size, samples, vifNo, VIF, Outliers, cut_off)
        #             sys.stdout = saveout
        #             fsock.close()
        #             f = open(output, "r")
        #             df = f.read()
        return HttpResponse("Results loaded")


class Pyspark_pipeline(APIView):
    def post(self, data):
        df = json.loads(data.body)
        json_args = Inputargs.objects.create(args=df)
        json_args.save()
        pd_df = pd.DataFrame.from_dict(df)
        dname, target, DTClassifier, RFClassifier, config_object_user, model_file, predict, phenome_data, user_defined_terminology, sample_type, description, uom_type, all_M, d_col, index, to_csv_name, Classification_or_Regression, analysis_id, output = argss(
            pd_df, df)
        saveout = sys.stdout
        fsock = open(output, 'w')
        sys.stdout = fsock
        Decision_M_Pyspark(dname, target, DTClassifier, RFClassifier, config_object_user, model_file, predict,
                           phenome_data, user_defined_terminology,
                           sample_type, description, uom_type, all_M, d_col, index, to_csv_name,
                           Classification_or_Regression, analysis_id, output)
        sys.stdout = saveout
        fsock.close()
        f = open(output, "r")
        df = f.read()
        return HttpResponse(f"Results={df}")
#         except Exception as e:
#             return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


# @api_view(["POST"])
# def ML_CMD(data):
#     try:
#         df = json.loads(data.body)
#         pd_df = pd.DataFrame.from_dict(df)
#         dname,target,sklearn,algorithm,config_object_user,ai ,model_file,predict,phenome_data,user_defined_terminology,sample_type,description,uom_type,all_M,d_col,index,to_csv_name,cut_off=args(pd_df,df)

#         Decision_M(dname,target,sklearn,algorithm,config_object_user,ai ,model_file,predict,phenome_data,
#                user_defined_terminology,sample_type,description,uom_type,all_M,d_col,index,to_csv_name,cut_off
#                  )
#         return JsonResponse("Models_created",safe=False)
#     except ValueError as e:
#         return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

#     serializer_class = EndpointSerializer
# obj = Endpoint.objects.all().first()
# print(obj)
# context= {
#     "title":obj.id,
#    'des':obj.owner
# }
# print(context)
# Decision_M()
# return render(request,"detail.html")
