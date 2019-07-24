
from .functions import computeType,PPICompute,sPPICompute
import numpy as np
import itertools


class KeggNode:
	
	computeMetapaths = computeType
	
	def isThisNode(nodeValue):
		return (isinstance(nodeValue,str) and nodeValue[0:3] == "hsa")

class ReactomeNode:
	
	computeMetapaths = computeType
	
	def isThisNode(nodeValue):
		return (isinstance(nodeValue,str) and nodeValue[0:2] == "R-")

class GoNode:
	
	computeMetapaths = computeType
	
	def isThisNode(nodeValue):
		return (isinstance(nodeValue,str) and nodeValue[0:3] == "GO:")

class InterproNode:
	
	computeMetapaths = computeType
	
	def isThisNode(nodeValue):
		return (isinstance(nodeValue,str) and nodeValue[0:3] == "IPR")
	
class ProteinInteractionNode:
	computeMetapaths = sPPICompute
	
	def isThisNode(nodeValue):
		#if nodeValue.isdigit():
		#	nodeValue = int(nodeValue)
		return (isinstance(nodeValue,int) or isinstance(nodeValue,np.integer) or nodeValue.isdigit())


#class StaticFeature():
	

