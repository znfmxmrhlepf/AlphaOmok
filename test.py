from tools import getOptions
from game import omok

opts = getOptions()
game = omok(opts)

done = False
rwd = 0

print(game.root.shape)
game.showImg()

while not done:
    i, j = input().split()
    action = [int(i), int(j)]
    done, rwd = game.step(action)

    print(rwd)