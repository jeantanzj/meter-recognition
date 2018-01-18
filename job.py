

import cv2
import performRecognition
import json
import time
from datetime import datetime
import os, errno
import argparse
import requests
import logger

tunables = ['threshold','minboxarea','invert','format','tolerance']

class Job(object):
    def __init__(self,args):
        args['classiferPath'] = 'classifiers/digits_cls_svhn_nobw.pkl'
        args['verbose'] = True
        args['mindigits'] =  args['format'].count('#')
        self.args = args
        self.server = args['server']
        self.camera_id = args['camera_id']
        self.output_dir = self.get_output_dirname()
        self.log = logger.SimpleLogger(self.output_dir)
        self.mac = self.log.get_mac()

    def reinitArgs(self):
        self.args = args
        self.output_dir = self.get_output_dirname()
        self.log = logger.SimpleLogger(self.output_dir)

    def get_parameters_from_server(self):
        url = '{}/tuning/{}'.format(self.server, self.mac)
        r = requests.get(url)
        print(r.status_code)
        print(r.content)
        params = json.loads(r.content)
        if len(params) >0 :
            params = params[0]
        return params

    def post_parameters_to_server(self):
        url = '{}/tuning/{}'.format(self.server, self.mac)
        data={'mac':self.mac}
        for param in tunables:
            data[param] = self.args[param]
        try:
            r = requests.post(url, data=data)
            if r.status_code != 200:
                self.log.error({"msg": "Failed to post_parameters_to_server", 
                    "params": {"status_code": r.status_code, "response": r.content, "item": data}})
        except:
            pass

    def get_image1(self):
        cam = cv2.VideoCapture(self.camera_id)
        s, im = cam.read()
        if s:
            h = min(300, len(im))
            w = int(float(h)/len(im) * len(im[0]))
            im= cv2.resize(im,(w, h))
            return im, self.log.get_timestamp()
        return None, self.log.get_timestamp()

    def get_image(self):
        im = cv2.imread('photos/gt.jpg')
        h = min(300, len(im))
        w = int(float(h)/len(im) * len(im[0]))
        im=cv2.resize(im,(w, h))
        return im, self.log.get_timestamp()

    def get_history(self):
        try:
            with open(os.path.join(self.output_dir, '{}.json'.format(self.mac)), 'r') as f:
                return json.load(f)
        except IOError as exc:
            if exc.errno != errno.ENOENT:
                raise
        return {}

    def write_to_history(self,history):
        with open(os.path.join(self.output_dir, '{}.json'.format(self.mac)), 'w') as f:
            json_str = json.dumps(history)
            f.write(json_str)


    def get_output_dirname(self):
        return 'm{}t{}b{}o{}r{}d{}'.format(
            self.args['camera_id'],int(self.args['threshold']), 
            self.args['minboxarea'], self.args['tolerance'], 
            int(self.args['invert']), self.args['mindigits'])

    def inform_server(self,item):
        data = item
        data['mac'] = self.mac
        data['cam_id'] = self.camera_id
        picture = open(data['filename'], 'rb')
        payload = {
            'json' : (None, json.dumps(data), 'application/json'),
            'file' : (os.path.basename(data['filename']), picture, 'application/octet-stream')
        }
        try:
            r = requests.post('{}/{}'.format(self.server,'update'), 
                files=payload, timeout = 2)
            if r.status_code != 200:
                self.log.error({"msg": "Failed to inform server ", 
                    "params": {"status_code": r.status_code, "response": r.text, "item": data}})
            
        finally:
            picture.close()
        


    def capture_and_calculate(self):

        trials = 1
        cnt = 1
        while cnt <= trials:
            im, time_taken = self.get_image()
            predictions = performRecognition.predict(im, self.args)
            if predictions is not None:
                val = []
                i = 0
                for d in self.args['format']:
                    if d == '#':
                        val.append(str(predictions[i]))
                        i+=1
                    elif d == '.':
                        val.append(d)

                try:
                    predictedValue = float(''.join(val))
                    filename = os.path.join(self.output_dir,str(time_taken)+'.jpg')
                    cv2.imwrite(filename, im)
                    item = {'time': datetime.fromtimestamp(time_taken).strftime('%Y-%m-%d %H:%M:%S'), 
                        'reading': predictedValue, 'filename': filename}
                    self.log.info(item)
                    history = self.get_history()
                    cid = str(self.camera_id)

                    if cid not in history.keys():
                        history[cid] = []

                    previous_item = None
                    if len(history[cid]) > 0 :
                        previous_item = history[cid][-1]
                        delta = item['reading'] - previous_item['reading']
                        item['delta'] = delta
                        print('delta', delta)

                    history[cid].append(item)
                    self.write_to_history(history)
                    self.inform_server(item)
                    return True
                except Exception as e:
                    cnt +=1
                    self.log.error({ "msg": str(e), "params": {"val": val} })
            else:
                self.log.info({"msg": "Failed to get reading from image. Trying again. ({}/{})".format(cnt, trials)})
                cnt += 1
                time.sleep(2)

        self.log.error({"msg": 'Exceeded number of trials without successful prediction'})
        return False
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", help="Threshold to remove noise (0-255)", type=performRecognition.threshold_float, 
        default=0.0)
    parser.add_argument("-r", "--invert", help="Invert the threshold", action="store_true")
    parser.add_argument("-b", "--minboxarea", help="Minimum area of rectangle to be considered digits", type=int, default=0)
    parser.add_argument("-o", "--tolerance", help="Tolerance of difference in length and height of digits", type=int, default=3)
    parser.add_argument("-m", "--camera_id", type=int, default=0)
    parser.add_argument("-f", "--format", type=str, default="#####")
    parser.add_argument("-s", "--server", type=str, default="http://127.0.0.1:8050")
    args = vars(parser.parse_args())
    job = Job(args)
    params = job.get_parameters_from_server()
    print('params',params)
    if params:
        for param in ['threshold','minboxarea','tolerance']:
            if params[param] is not None:
                job.args[param] = int(params[param])
        for param in ['mac','format']:
            if params[param] is not None:
                job.args[param] = str(params[param])
        if params['invert'] is not None:
            job.args['invert'] = bool(params[param])
        job.reinitArgs()
        job.log.info({"msg": 'Using params from server', "params": job.args})
        
    else:
        job.post_parameters_to_server()
    
    job.capture_and_calculate()