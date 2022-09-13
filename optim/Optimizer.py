import numpy as np

# SGD optimizer is assumed

class SGD:
    def __init__(self):
        pass

    def update(self, w, grad, lr):
        """
        [Inputs]
            w : current weight
            grad : gradient for w
            lr : learning rate

        [Outputs]
            updated_weight : updated weight.
        """
        updated_weight = w - lr * grad

        return updated_weight
