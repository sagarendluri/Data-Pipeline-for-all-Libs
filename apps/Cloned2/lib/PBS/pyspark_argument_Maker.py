import sys
def argss(pd_df,df):
    def dname():
        dname = "dname"
        if dname in pd_df.columns:
            dname =df[0].get(dname)
            return dname
        else:
            return None
    d_name=dname()
    def target():
        target = "target"
        if target in pd_df.columns:
            target =df[0].get(target)
            return target
        else:
            return None
    target_=target()
    def DTClassifier():
            DTClassifier = "DTClassifier"
            if DTClassifier in pd_df.columns:
                DTClassifier =df[0].get(DTClassifier)
                return DTClassifier
            else:
                return None
    DTClassifier=DTClassifier()
    def RFClassifier():
        RFClassifier="RFClassifier"
        if RFClassifier in pd_df.columns:
            RFClassifier =df[0].get(RFClassifier)
            return RFClassifier
        else:
            return None
    RFClassifier=RFClassifier() 
    def config_object_user():
        config_object_user = "config_object_user"
        if config_object_user in pd_df.columns:
            config_object_user =df[0].get(config_object_user)
            return config_object_user
        else:
            return None
    config_object_user_=config_object_user()
    # print(ai)
    def model_file():
        model_file ="model_file"
        if model_file in pd_df.columns:
            amodel_filei =df[0].get(model_file)
            return model_file
        else:
            return None
    model_file_=model_file() 
    def predict():
        predict ="predict"
        if predict in pd_df.columns:
            predict =df[0].get(predict)
            return predict
        else:
            return None
    predict_=predict() 

    def phenome_data():
        phenome_data ="phenome_data"
        if phenome_data in pd_df.columns:
            phenome_data =df[0].get(phenome_data)
            return phenome_data
        else:
            return None
    phenome_data_=phenome_data() 

    def user_defined_terminology():
        user_defined_terminology ="user_defined_terminology"
        if user_defined_terminology in pd_df.columns:
            user_defined_terminology =df[0].get(user_defined_terminology)
            return user_defined_terminology
        else:
            return None
    user_defined_terminology_=user_defined_terminology()
    def sample_type():
        sample_type ="sample_type"
        if sample_type in pd_df.columns:
            sample_type =df[0].get(sample_type)
            return sample_type
        else:
            return None
    sample_type_=sample_type() 
    def description():
        description ="description"
        if description in pd_df.columns:
            description =df[0].get(description)
            return description
        else:
            return None
    description_=description() 
    def uom_type():
        uom_type ="uom_type"
        if uom_type in pd_df.columns:
            uom_type =df[0].get(uom_type)
            return uom_type
        else:
            return None
    uom_type_=uom_type()
    def all_M():
        all_M ="all_M"
        if all_M in pd_df.columns:
            all_M =df[0].get(all_M)
            return all_M
        else:
            return None
    all_M_=all_M()
    def d_col():
        d_col ="d_col"
        if d_col in pd_df.columns:
            d_col =df[0].get(d_col)
            return d_col
        else:
            return None
    d_col_=d_col() 
    def index():
        index ="index"
        if index in pd_df.columns:
            index =df[0].get(index)
            return index
        else:
            return None
    index_=index()
    def to_csv_name():
        to_csv_name ="to_csv_name"
        if to_csv_name in pd_df.columns:
            to_csv_name =df[0].get(to_csv_name)
            return to_csv_name
        else:
            return None
    to_csv_name_=to_csv_name()  
    def Classification_or_Regression():
        Classification_or_Regression ="Classification_or_Regression"
        if Classification_or_Regression in pd_df.columns:
            Classification_or_Regression =df[0].get(Classification_or_Regression)
            return Classification_or_Regression
        else:
            return 0.8
    Classification_or_Regression=Classification_or_Regression()
    def analysis_id():
        analysis_id ="analysis_id"
        if analysis_id in pd_df.columns:
            analysis_id =df[0].get(analysis_id)
            return analysis_id
        else:
            print("analysis_id IS not given")
            sys.exit()
            return None
    analysis_id=analysis_id()
    def output():
        output ="output"
        if output in pd_df.columns:
            output =df[0].get(output)
            return output
        else:
            return "test_"
    output=output()
    print(d_name,target_,DTClassifier,RFClassifier,config_object_user_ ,model_file_,predict_,phenome_data_,user_defined_terminology_,sample_type_,description_,uom_type_,all_M_,d_col_,index_,to_csv_name_,Classification_or_Regression,analysis_id,output)
    return  d_name,target_,DTClassifier,RFClassifier,config_object_user_ ,model_file_,predict_,phenome_data_,user_defined_terminology_,sample_type_,description_,uom_type_,all_M_,d_col_,index_,to_csv_name_,Classification_or_Regression,analysis_id,output
                