import json
from uuid import getnode
from datetime import datetime
import time
import os,errno

class Utils(object):
    def __init__(self):
        mac = getnode()
        if (mac >> 40) % 2 :
            raise OSError('Invalid mac address {}'.format(hex(mac)))
        else:
            self.__mac = hex(mac)

    def get_timestamp(self):
        return time.time()

    def get_time_as_string(self):
        return datetime.fromtimestamp(self.get_timestamp()).strftime('%Y-%m-%d %H:%M:%S')

    def get_mac(self):
        return self.__mac


class SimpleLogger(Utils):
    def __init__ (self, output_dir):
        self.output_dir = output_dir
        try:
            os.mkdir(self.output_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        super(SimpleLogger, self).__init__()

    def error(self,message):
        with open(os.path.join(self.output_dir,'error.log'),'a') as f:
            json_str = json.dumps({"id": "{}".format(self.get_mac()) ,"type":"error","time": self.get_time_as_string(), "message": message})
            f.write(json_str+"\n")

    def info(self,message):
        with open(os.path.join(self.output_dir,'info.log'),'a') as f:
            json_str = json.dumps({"id": "{}".format(self.get_mac()) , "type":"info","time": self.get_time_as_string(), "message": message})
            f.write(json_str+"\n")