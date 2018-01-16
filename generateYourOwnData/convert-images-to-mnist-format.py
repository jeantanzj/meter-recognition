import os
from PIL import Image
from array import *
from random import shuffle
import json
#Photos need to be 28*28
#Run `sips -z 28 28 folder/*/*.png ` to resize
#convert from format to jpg
#for i in *.png; do sips -s format jpeg -s formatOptions 70 "${i}" --out "${i%png}jpg"; done 

# Load from and save to

#Names = [['../output_square','train'], ['./test-images','test']]
Names = [['../output_square','train-nobw']]
total = 0
for name in Names:
	
	data_image = array('B')
	data_label = array('B')

	FileList = []
	for dirname in os.listdir(name[0])[1:]: # [1:] Excludes .DS_Store from Mac OS

		path = os.path.join(name[0],dirname)
	
		for filename in os.listdir(path):
			if filename.endswith(".png") :
				FileList.append(os.path.join(path, filename))

	#shuffle(FileList) # Usefull for further segmenting the validation set
	errors = []

	for filename in FileList:
		try:
			label = int(os.path.basename(os.path.abspath(os.path.join(filename,os.path.pardir))))
			f = os.path.abspath(filename)
			print(f)
			Im = Image.open(f)
			#Im = Im.convert('1')
			pixel = Im.load()

			width, height = Im.size

			for x in range(0,width):
				for y in range(0,height):
					data_image.append(pixel[y,x])

			data_label.append(label) # labels start (one unsigned byte each)
			total +=1
		except Exception as e:
			errors.append(json.dumps({"err": str(e), "fn": filename}))

	if len(errors) > 0:
		print( len(errors) , "occurred")
		with open("errors.json",'w') as errfile:
			errfile.write(json.dumps(errors))	

	#hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
	hexval = "{0:#0{1}x}".format(len(FileList),10) # number of files in HEX
	# header for label array

	header = array('B')
	#header.extend([0,0,8,1,0,0])
	#header.append(long('0x'+hexval[2:][:2],16))
	#header.append(long('0x'+hexval[2:][2:],16))
	header.extend([0,0,8,1])
	header.append(int('0x'+hexval[2:][:2],16))
	header.append(int('0x'+hexval[4:][:2],16))
	header.append(int('0x'+hexval[6:][:2],16))
	header.append(int('0x'+hexval[8:][:2],16))

	data_label = header + data_label

	# additional header for images array
	
	# if max([width,height]) <= 256:
	# 	header.extend([0,0,0,width,0,0,0,height])
	# else:
	# 	raise ValueError('Image exceeds maximum size: 256x256 pixels');
	hexval = "{0:#0{1}x}".format(width,10) # width in HEX
	header.append(int('0x'+hexval[2:][:2],16))
	header.append(int('0x'+hexval[4:][:2],16))
	header.append(int('0x'+hexval[6:][:2],16))
	header.append(int('0x'+hexval[8:][:2],16))
	hexval = "{0:#0{1}x}".format(height,10) # height in HEX
	header.append(int('0x'+hexval[2:][:2],16))
	header.append(int('0x'+hexval[4:][:2],16))
	header.append(int('0x'+hexval[6:][:2],16))
	header.append(int('0x'+hexval[8:][:2],16))


	header[3] = 3 # Changing MSB for image data (0x00000803)
	
	data_image = header + data_image

	output_file = open(name[1]+'-images-idx3-ubyte-'+str(total), 'wb')
	data_image.tofile(output_file)
	output_file.close()

	output_file = open(name[1]+'-labels-idx1-ubyte-'+str(total), 'wb')
	data_label.tofile(output_file)
	output_file.close()

# gzip resulting files

for name in Names:
	os.system('gzip '+name[1]+'-images-idx3-ubyte-'+str(total))
	os.system('gzip '+name[1]+'-labels-idx1-ubyte-'+str(total))
print('total', total)