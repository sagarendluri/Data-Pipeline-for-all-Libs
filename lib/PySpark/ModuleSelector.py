import sys
from PySpark_DTreeClassifier import DTreeClassifier
from Pyspark_NBClassifier import NBClassifier
from Pyspark_RFClassifier import RFClassifier
from Pyspark_MLPClassifier import MLPClassifier
from getFeatureVector_DT_RF import spark
from getFeatureVector_ML_NB import Spark_
class modularization():
    def __init__(self,dname,i,drop_col,Classification_or_Regression,DTC,NBC,RFC,MLPC,ALL,config_object_user,
                         user_defined_terminology,sample_type ,description ,uom_type):
        self.dname = dname
        self.i = i 
        self.drop_col =drop_col
        self.Classification_or_Regression=Classification_or_Regression
        self.DTC = DTC
        self.NBC = NBC
        self.RFC = RFC
        self.MLPC = MLPC
        self.ALL = ALL
        self.config_object_user = config_object_user
        self.user_defined_terminology = user_defined_terminology
        self.sample_type =sample_type
        self.description = description
        self.uom_type = uom_type
        self.Selection()
    def DTreeRF(self,final,df):
        if "DTClassifier" == self.DTC:
            print("DTC")
            Model = DTreeClassifier(final,df,self.dname,self.i,self.Classification_or_Regression,self.config_object_user,
                         self.user_defined_terminology,self.sample_type ,self.description ,self.uom_type,self.drop_col)
            Model.building()
        elif "RFClassifier" == self.RFC:
            print("RFC")
            Model = RFClassifier(final,df,self.dname,self.i,self.Classification_or_Regression,self.config_object_user,
                         self.user_defined_terminology,self.sample_type ,self.description ,self.uom_type,self.drop_col)
            Model.building()
    def MLP_NB(self,final,o_n ,i_n,vectorAssembler,indexer,input_nodes):
        if "NBClassifier" == self.NBC:
            print("NBC")
            Model = NBClassifier(input_nodes,vectorAssembler,indexer)
            Model.building()
        elif "MLPClassifier" == self.MLPC:
            Model = MLPClassifier(final,o_n ,i_n)
            Model.building()
            print("MLPC")
    def Selection(self):
        if "MLPClassifier" == self.MLPC or "NBClassifier" == self.NBC:
            print("Nice")
            clean = Spark_(self.dname,self.i)
            clean.models()
            self.o_n ,self.i_n , self.input_nodes = clean.nodes()
            self.final ,self.vectorAssembler,self.indexer = clean.data_nodes()
            self.MLP_NB(self.final,self.o_n ,self.i_n,self.vectorAssembler,self.indexer,self.input_nodes)
        else: 
            clean = spark(self.dname,self.i,self.drop_col)
            clean.models()
            self.final,self.df=clean.cleanDataFrame()
            self.DTreeRF(self.final,self.df)
    def all_modules(self):
        if "ALL_Modules"==self.ALL:
            clean = spark(self.dname,self.i)
            clean.models()
            final=clean.cleanDataFrame()
            Model = DTreeClassifier(final)
            Model.building()
            Model = RFClassifier(final)
            Model.building()
            clean = Spark_(self.dname,self.i)
            clean.models()
            o_n ,i_n , input_nodes = clean.nodes()
            final ,vectorAssembler,indexer = clean.data_nodes()
            Model = NBClassifier(input_nodes,vectorAssembler,indexer)
            Model.building()
            Model = MLPClassifier(final,o_n ,i_n)
            Model.building()
            
        
        