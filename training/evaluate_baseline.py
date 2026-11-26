import sys

from evaluate_varsha import main


if __name__ == "__main__":
    main(["--baselines-only", *sys.argv[1:]])
