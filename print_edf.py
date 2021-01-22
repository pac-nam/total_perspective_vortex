import argparse
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="define your file", type = str)
	parser.add_argument("-s", "--size", help="seconds showed. Default = 30", type = float, default=30)
	args = parser.parse_args()
	raw = read_raw_edf(args.file, preload=True, stim_channel='auto', verbose=False)
	raw.plot(duration=args.size, block=True, scalings='auto')
	plt.show()

if __name__ == "__main__":
	main()