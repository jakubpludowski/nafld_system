import argparse


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="nafld")
    parser.add_argument("--mode", required=True, type=str, dest="mode")
    args, _ = parser.parse_known_args()
    return args
