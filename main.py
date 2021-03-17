from vortex import vortex
import argparse

def run(args) :
    if (args.training is True and args.predict is True) or (args.training is False and args.predict is False) :
        print("Run training or predict")
        return(0)
    vort = vortex(args)
    if args.training is True :
        vort.training()
    if args.predict is True :
        vort.predict()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--visualize", help="plot data", action="store_true")
    parser.add_argument("-p","--path", help="define the path for mne save", type = str, default=None)
    parser.add_argument("-t","--training", help="train the model", action="store_true")
    parser.add_argument("-pr","--predict", help="Run predict", action="store_true")
    args = parser.parse_args()
    run(args)