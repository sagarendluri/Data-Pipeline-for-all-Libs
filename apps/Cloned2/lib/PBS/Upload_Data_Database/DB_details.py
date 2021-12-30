from configparser import ConfigParser
import pymysql
import sqlalchemy
class DB_Credentials():
    def __init__(self,config_object_user,db_name):
        self.config_object_user =config_object_user
        self.db_name = db_name
    def DB_details(self):
        if "Trials_DB" == self.db_name:
            config_object = ConfigParser()
            ini = config_object.read(r'/dse/apps/Cloned2/config.ini')#C:\Users\sagar\dse_sagar\config.ini')   #r'/dse/apps/Cloned2/config.ini')
            config_object.read(ini)
            userinfo = config_object[self.db_name]
            password= userinfo["password"]
            user= userinfo["user"]
            host= userinfo["host"]
            db= userinfo["db"]
            conn = pymysql.connect(host=(', '.join(["%s" % host])),
                                   max_allowed_packet = 1073741824,
                       port=int(3306),
                       user=(', '.join(["%s" % user])),
                       passwd=(', '.join(["%s" % password])),
                       db=(', '.join(["%s" % db])), charset='utf8mb4')
            print("db connection success")
            cursor=conn.cursor()
            return cursor ,conn
        else:
            config_object = ConfigParser()
            ini = config_object.read(r'/dse/apps/Cloned2/config.ini')#C:\Users\sagar\dse_sagar\config.ini')   #r'/dse/apps/Cloned2/config.ini')
            config_object.read(ini)
            userinfo = config_object[self.config_object_user]
            password= userinfo["password"]
            user= userinfo["user"]
            host= userinfo["host"]
            db= userinfo["db"]
            conn = pymysql.connect(host=(', '.join(["%s" % host])),
                                   max_allowed_packet = 1073741824,
                       port=int(3306),
                       user=(', '.join(["%s" % user])),
                       passwd=(', '.join(["%s" % password])),
                       db=(', '.join(["%s" % db])), charset='utf8mb4')
            print("db connection success")
            cursor=conn.cursor()
            return cursor ,conn
    