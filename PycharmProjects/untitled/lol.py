from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import cv2
import numpy


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

    t = loadImage('krests/krest1.png')

    net = buildNetwork(len(t), len(t), 1)
    dataset = SupervisedDataSet(len(t), 1)
    dataset.addSample(loadImage('krests/krest1.png'), 100)
    dataset.addSample(loadImage('krests/krest2.png'), 100)
    dataset.addSample(loadImage('krests/krest3.png'), 100)
    dataset.addSample(loadImage('krests/krest4.png'), 100)
    dataset.addSample(loadImage('krests/krest5.png'), 100)
    dataset.addSample(loadImage('nols/nol1.png'), -100)
    dataset.addSample(loadImage('nols/nol2.png'), -100)
    dataset.addSample(loadImage('nols/nol3.png'), -100)
    dataset.addSample(loadImage('nols/nol4.png'), -100)
    dataset.addSample(loadImage('nols/nol5.png'), -100)
    dataset.addSample(loadImage('nols/nol6.png'), -100)
    dataset.addSample(loadImage('nols/nol7.png'), -100)
   #dataset.addSample(loadImage('stable/bullet1.png'), 0)


    trainer = BackpropTrainer(net, dataset)
    error = 10
    iteration = 0
    while error > 0.001:
        error = trainer.train()
        iteration += 1
        print("Error on iteration {0} is {1}".format(iteration, error))

    print("\nResult for 'krest.png': ", net.activate(loadImage('krests/krest.png')))
    print("\nResult for 'nol.png': ", net.activate(loadImage('nols/nol.png')))