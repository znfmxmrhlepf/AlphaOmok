from tools import getOptions
from game import omok

opts = getOptions()
game = omok(opts)

done = False
rwd = 0

game.createWindow()

while not done:
    i, j = input().split()
    act = [int(i), int(j)]
    done, rwd = game.step(act)

    print(rwd)