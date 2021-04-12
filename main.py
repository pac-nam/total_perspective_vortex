from srcs.vortex import Vortex
import argparse

def run(args) :
    if (args.training is True and args.predict is True) or (args.training is False and args.predict is False) :
        print("Run training or predict")
        return(0)
    if (args.subject < 1 or args.subject > 109) :
        print("Subject must be between 1 and 109 included")
        return(0)
    vort = Vortex(args)
    data = [[3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]]
    score = 0
    if args.training is True :
        for i in range(len(data)) :
            score += vort.training(data[i], i)
        mean = score / 4
        print("Mean score : {}%".format(round(mean * 100, 2)))
    if args.predict is True :
        for i in range(len(data)) :
            score += vort.predict(data[i], i)
        mean = score / 4
        print("Mean score : {}%".format(round(mean, 2)))
        if mean < 50 :
            print("\033[93mWARNING : Have you used the same subject for predict and training ?\033[0m")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--visualize", help="plot data", action="store_true")
    parser.add_argument("-p","--path", help="define the path for mne save", type = str, default=None)
    parser.add_argument("-s","--subject", help="define the subject number [1-109]", type = int, default=1)
    parser.add_argument("-t","--training", help="train the model", action="store_true")
    parser.add_argument("-pr","--predict", help="run predict", action="store_true")
    args = parser.parse_args()
    run(args)