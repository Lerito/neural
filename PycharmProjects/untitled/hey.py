import cv2
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def loadImage(path):
    im = cv2.imread(path)
    return flatten(im)


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

if __name__ == "__main__":

    t = loadImage('stable/Square.png')

    net = buildNetwork(len(t), len(t), 1)
    dataset = SupervisedDataSet(len(t), 1)
    dataset.addSample(loadImage('stable/Square.png'), 100)
    dataset.addSample(loadImage('stable/almostguy.png'), 100)
    dataset.addSample(loadImage('stable/pants.png'), 100)
    dataset.addSample(loadImage('stable/chemestry.png'), 100)
    dataset.addSample(loadImage('stable/Tetris.png'), 100)
    dataset.addSample(loadImage('stable/rectangle.png'), 100)
    dataset.addSample(loadImage('stable/stableTriangle.png'), 100)
    dataset.addSample(loadImage('stable/SquareSmall.png'), 100)

    dataset.addSample(loadImage('almoststable/diamond.png'), 0)
    dataset.addSample(loadImage('almoststable/T.png'), 0)

    dataset.addSample(loadImage('unstable/whistle.png'), -100)
    dataset.addSample(loadImage('unstable/triagle.png'), -100)
    dataset.addSample(loadImage('unstable/thumbdown.png'), -100)
    dataset.addSample(loadImage('unstable/stick.png'), -100)
   #dataset.addSample(loadImage('stable/bullet1.png'), 0)


    trainer = BackpropTrainer(net, dataset)
    error = 10
    iteration = 0
    while error > 0.000000000001:
        error = trainer.train()
        iteration += 1
        print("Error on iteration {0} is {1}".format(iteration, error))

    print("\nResult for 'bullet1.png': ", net.activate(loadImage('stable/bullet1.png')))
    print("\nResult for 'bullet.png': ", net.activate(loadImage('stable/bullet.png')))
    print("\nResult for 'lemon.png': ", net.activate(loadImage('almoststable/lemon.png')))
