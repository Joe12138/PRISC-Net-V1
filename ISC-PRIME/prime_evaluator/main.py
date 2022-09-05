import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
from prime_evaluator.predictor import Predictor
from prime_evaluator.utils.parsing import parse_arguments


def main():
    """ Main """
    args = parse_arguments()
    predictor = Predictor(args)
    predictor.start()


if __name__ == "__main__":
    main()
