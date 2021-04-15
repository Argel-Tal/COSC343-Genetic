import tkinter as tk
from tkinter import filedialog
import sys, getopt
from game import Game

def main(argv):
    # Load the defaults
    from defaults import game_params

    # Check of arguments from command line
    try:
        opts, args = getopt.getopt(argv, "r:f:l:",["res=", "fast=", "load="])
    except getopt.GetoptError:
        print("Error! Invalid argument.")
        sys.exit(2)

    # Process command line arguments
    loadGame = None
    for opt, arg in opts:
        if opt in ("-r", "--res"):
            res = arg.split('x')
            if len(res) != 2:
               print("Error! The -r/res= argument must be followed with <width>x<height> specifiction of resolution (no spaces).")
               sys.exit(-1)

            game_params['visResolution'] = (int(res[0]), int(res[1]))
        elif opt in ("-f", "--fast"):
            game_params['visSpeed'] = arg

        elif opt in ("-l", "--load"):
            loadGame = arg

    if game_params['visSpeed'] != 'normal' and game_params['visSpeed'] != 'fast' and game_params['visSpeed'] != 'slow':
        print("Error! Invalid setting '%s' for visualisation speed.  Valid choices are 'slow','normal',fast'" % game_params['visSpeed'])
        sys.exit(-1)

    if loadGame is None:
       # If load game wasn't specified in the command line arguments then
       # open a dialog box in the 'saved' folder
       root = tk.Tk()
       root.withdraw()

       loadGame = filedialog.askopenfilename(initialdir="../Submissions_2020")

    # Load a previously saved game
    Game.load(loadGame,visResolution=game_params['visResolution'],
               visSpeed=game_params['visSpeed'])


if __name__ == "__main__":
   main(sys.argv[1:])