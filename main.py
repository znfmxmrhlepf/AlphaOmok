from train import trainer
import tools
from game import omok

opt = tools.getOptions()
env = omok(opt)
trainer(env, opt)