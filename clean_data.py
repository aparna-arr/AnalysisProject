#!/share/software/user/open/python/3.6.1/bin/python3
import sys

if len(sys.argv) < 3:
	print("usage: clean_data.py <downloaded.csv> <output_name.tsv>\n", file = sys.stderr)
	sys.exit(1)

downloadFile = sys.argv[1]
outputFilename = sys.argv[2]

dfh = open(downloadFile, "r")
dout = open(outputFilename, "w")

dout.write("ChrIndex\tBarcode\tx\ty\tz\n")

firstLine = True

for line in dfh:
	if (firstLine == True):
		firstLine = False
		continue

	elem = line.rstrip().split(',')

	if not elem[0].isdigit():
		continue

	chr_index = elem[0]
	barcode_index = elem[1] # Bogdan calls these "segments"
	z = elem[2]
	x = elem[3]
	y = elem[4]
		
	dout.write(chr_index + "\t" + barcode_index + "\t" + x + "\t" + y + "\t" + z + "\n")
	
dout.close()
dfh.close()
