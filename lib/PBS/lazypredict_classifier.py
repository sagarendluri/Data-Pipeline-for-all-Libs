
from QC import train_test
from Fill_DB import DB_upload
from Fill_h2o_DB import h2o_DB_upload
import DB_details
from lazypredict.Supervised import LazyClassifier
def lazypredict_classifier(data, y,i, dname,config_object_user,
                                 user_defined_terminology,sample_type ,description ,uom_type,cut_off):
    X = self.data_sorted.drop([i],axis=1)
    print(X.shape)
    Y = self.data_sorted[self.i]
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
    X_train, X_test = train_test(X_train, X_test,cut_off)
    cls= LazyClassifier(ignore_warnings=False, custom_metric=None)
    models, predictions = cls.fit(X_train, X_test, y_train, y_test)