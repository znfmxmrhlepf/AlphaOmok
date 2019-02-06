import argparse

def getOptions():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--GAME_SIZE", type=int, default=19, help="size of board")
    parser.add_argument("--MAX_LENGTH", type=int, default=5, help="max length of stone")
    parser.add_argument("--SHOW_IMG", type=bool, default=False)

    options = parser.parse_args()

    return options