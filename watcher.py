import logger
import requests

class Watcher(object):
	def __init__(self, url):
		self.log = logger.SimpleLogger('watcher')
		self.url = url
	def watch(self):
		r = requests.get(url, params={self.get_mac()})
		if r.status_code == 200:
			self.handle(r.json)
	def handle(self, js):
		pass
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-u','url',required=True)
	args = vars(parser.parse_args())
	w = Watcher(args.url)
	w.watch()