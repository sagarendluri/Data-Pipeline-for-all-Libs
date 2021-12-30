from celery import shared_task
from time import sleep
from apps.Cloned2.lib.PBS.arguments_Maker import args
from apps.Cloned2.lib.PBS.pyspark_argument_Maker import argss
from apps.Cloned2.bin.DSEMainPipeline import Decision_M


# @shared_task(bind = True)
def got_to_s(dname, target, sklearn, algorithm, config_object_user, ai, model_file, predict, phenome_data,
             user_defined_terminology, sample_type, description, uom_type, all_M, d_col, index, to_csv_name, min_depth,
             max_depth, min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors, xgb_objective,
             xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_C, svm_gamma, svm_kernel, default, analysis_id,
             db_name, N_features, filter_M, pca,
             learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping, dataset_type,
             test_size, samples, vifNo, VIF, Outliers, cut_off):
    Decision_M(dname, target, sklearn, algorithm, config_object_user, ai, model_file, predict, phenome_data,
               user_defined_terminology, sample_type, description, uom_type, all_M, d_col, index, to_csv_name,
               min_depth, max_depth, min_samples_split, n_estimators_start, n_estimators_stop, n_neighbors,
               xgb_objective, xgb_learning_rate, xgb_max_depth, xgb_min_child_weight, svm_C, svm_gamma, svm_kernel,
               default, analysis_id, db_name, N_features, filter_M, pca,
               learning_rate_init, max_iter, hidden_layer_sizes, activation, alpha, early_stopping, dataset_type,
               test_size, samples,  vifNo, VIF, Outliers, cut_off)

    return "task_Done"
