from .functions import computeType, sPPICompute
import numpy as np


class KeggNode:

    computeMetapaths = computeType

    def isThisNode(nodeValue):
        return isinstance(nodeValue, str) and nodeValue[0:3] == "hsa"


class ReactomeNode:

    computeMetapaths = computeType

    def isThisNode(nodeValue):
        return isinstance(nodeValue, str) and nodeValue[0:2] == "R-"


class GoNode:

    computeMetapaths = computeType

    def isThisNode(nodeValue):
        return isinstance(nodeValue, str) and nodeValue[0:3] == "GO:"


class InterproNode:

    computeMetapaths = computeType

    def isThisNode(nodeValue):
        return isinstance(nodeValue, str) and nodeValue[0:3] == "IPR"


class ProteinInteractionNode:
    computeMetapaths = sPPICompute

    def isThisNode(nodeValue):
        return (
            isinstance(nodeValue, int)
            or isinstance(nodeValue, np.integer)
            or nodeValue.isdigit()
        )
