'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from ConversionUtil import toScala
from random import randint,uniform,seed,choice,random

seed()

#Start at space and end at ~.
CHARMIN=32
CHARMAX=127

#Maximum value that the absolute value of a randomly generated float will be.
MAX_ABS_DOUBLE = 100

#The maximum length that a "long" string will be allowed to be.
#"Long" strings come from Pickle serialization terminology and refer to strings with length 
#greater than 255, i.e., whose length requires more than 1 byte to represent.
#Such long strings must be tested to ensure the validity of RDD conversions.
MAX_STRING_LENGTH=500

#Maximum length of strings that containers will be allowed to have.
#Should be kept to a minimum to avoid pain while debugging.
REALLY_SHORT_MAX_STRING_LENGTH=3

#Maximum length that a randomly generated tuple will be.
#Tuples of length 0 or greater than 22 are not tested because their use cases are virtually nonexistant,
#and some conversions regarding these sizes are not implemented.
MAX_TUPLE_LENGTH = 22

#Maximum length that a randomly generated List will be.
#Given that large Lists will be difficult to debug, this should be small.
MAX_LIST_LENGTH = 3

#Maximum length that a randomly generated Dict will be.
#Given that large Dicts will be difficult to debug, this should be small.
MAX_DICT_LENGTH = 3

#Maximum recursive "depth" allowed for containers that will have other containers as elements.
#A depth of 1 forbids a container to have another container as an element, 
#a depth of 2 allows a container to have a container with no containers as elements,
#a depth of 3 allows a container to have a container with containers with no containers as elements, and so on.
MAX_DEPTH = 3

#The frequency by which a recursive (non-simple) container will contain another container.
RECURSE_RATE = 0.1

'''
The following "Generation Functions" randomly generate many Python objects and also return their
Scala equivalents. In many cases, Py4J converts them automatially and toScala is not called.
However, it is an unfortunate problem that there is great difficultly in generating the same
random object in both Scala and Python without using some kind of conversion routine, which
is regrettably exactly what is being tested. Thus, in many cases, the to-be-tested method
"toScala" is used to acquire the equivalent Scala object.
'''

#Returns a random signed int/long with the specifying number of bits.
#Includes the entire 2's complement range for this number of bits.
def randInt(bits):
    return randint(-(1<<(bits-1)),(1<<(bits-1))-1)

#Returns a random signed 8-bit int and its Scala equivalent.
#This will be in the range of a Scala Byte.
def randInt8():
    num=randInt(8)
    return (num,num)

#Returns a random signed 16-bit int and its Scala equivalent.
#This will be in the range of a Scala Short.
def randInt16():
    num=randInt(16)
    return (num,num)

#Returns a random signed 32-bit int and its Scala equivalent.
#This will be in the range of a Scala Int.
def randInt32():
    num=randInt(32)
    return (num,num)

#Returns a random signed 64-bit int and its Scala equivalent.
#This will be in the range of a Scala Long.
def randLong64():
    num=randInt(64)
    #Need to convert to long because some Python ints may not fit into a Scala Int.
    #This is important for when toArray tries to implicitly determine the type of the array.
    return (long(num),toScala(num))

#Returns a random signed 128-bit int and its Scala equivalent.
#This will almost always be outside the range of a Scala long,
#and it is thus used to test Python int/long to scala.math.BigInt conversion validity.
def randLong128():
    num=randInt(128)
    return (long(num),toScala(num))

#Returns a random double in the range [-MAX_ABS_DOUBLE,MAX_ABS_DOUBLE] and its Scala equivalent.
def randDouble():
    doub = uniform(-MAX_ABS_DOUBLE,MAX_ABS_DOUBLE)
    return (doub,doub)

#Returns a random bool value and its Scala equivalent.
def randBoolean():
    if randint(0,1) == 1:
        return (True,True)
    return (False,False)

#Returns a random str with at most the specified length and its Scala equivalent.
def makeRandString(length):
    string = ""
    for i in range(length):
        string += chr(randint(CHARMIN,CHARMAX))
    return (string,string)

#Returns a random str with length at most 255 and its Scala equivalent.
def randShortByteString():
    return makeRandString(randint(0,255))

#Returns a random str with length at least 256 and at most MAX_STRING_LENGTH and its Scala equivalent.
def randLongByteString():
    return makeRandString(randint(256,MAX_STRING_LENGTH))

#Returns a random unicode string with length at most the specified length and its Scala equivalent.
#Only 1-2 byte encodings are tested due to apparent inconsistencies in how
#Python and Java encode with UTF-8 with larger encodings.
def makeRandUTF8String(length):
    string = unicode("")
    for i in range(length):
        if randint(0,1) == 1:
            string+=chr(randint(CHARMIN,CHARMAX))
        else:
            string+=unichr(randint(0x80,0x7FF))
    return (string,string)

#Returns a random str with length at most REALLY_SHORT_MAX_STRING_LENGTH and its Scala equivalent.
#Used to put strs in containers that will not cause the container representation to become unwieldy.
def randReallyShortString():
    return makeRandString(randint(0,REALLY_SHORT_MAX_STRING_LENGTH))

#Returns a random unicode string with length at most 255 and its Scala equivalent.
def randShortUTF8String():
    return makeRandUTF8String(randint(0,255))

#Returns a random unicode string with length at least 256 and
#at most MAX_STRING_LENGTH and its Scala equivalent.
def randLongUTF8String():
    return makeRandUTF8String(randint(256,MAX_STRING_LENGTH))

#Simply returns a 2-tuple containing Nones. Py4J should automatically convert
#a None to a null if called in a Java context.
def putNone():
    return (None,None)


#The simpleTypePool consists of all to-be-tested non-container types which "behave nicely" in Py4J when interacting with Scala containers.
#By "behave nicely", I mean that either they are not an implicitly converted Py4J type, or if they are, then the default conversion of that
#type into a Java type is a correct and well-formed conversion for ALL members of that type.
#For example, bool converts implicitly to Boolean correctly for both True/False, so no problem there.
#64-bit longs will convert implicitly to Scala longs, so no problem there either.
#However, >64-bit longs will NOT implicitly convert to scala.BigInt, even though the reverse is supported.
#Whats worse is that Py4J will try to implicitly convert such a value into a java long whenever some operation requires a Java object,
#particularly container assignment. Py4J will then raise an exception due to the failed conversion.
#Finally, doubles are also omitted from the pool due to the tediousness of ensuring that a converted double differs an acceptably low amount
#from its pre-converted partner.
simpleTypePool=[randInt16,randInt32,randLong64,randBoolean,randReallyShortString,putNone]

#Returns a random tuple and its Scala equivalent whose elements are generated randomly from simpleTypePool. 
def randSimpleTuple():
    pyList = []
    length = randint(1,MAX_TUPLE_LENGTH)
    for i in range(length):
        pyList.append(choice(simpleTypePool)()[0])
    tup = tuple(pyList)
    return (tup,toScala(tup))

#Returns a random list and its Scala equivalent whose elements are generated randomly from simpleTypePool. 
#The generated list has length at most MAX_LIST_LENGTH.
def randSimpleList():
    pyList = []
    length = randint(0,MAX_LIST_LENGTH)
    for i in range(length):
        pyList.append(choice(simpleTypePool)()[0])
    return (pyList,toScala(pyList))

#Returns a random dict and its Scala equivalent whose elements are generated randomly from simpleTypePool. 
#The generated dict has length at most MAX_DICT_LENGTH.        
def randSimpleDict():
    pyDict = {}
    length = randint(0,MAX_DICT_LENGTH)
    for i in range(length):
        pyDict[choice(simpleTypePool)()[0]]=choice(simpleTypePool)()[0]
        #Overwriting keys isn't really a big deal, since we are choosing the length randomly anyway.
    return (pyDict,toScala(pyDict))

#Pool comprising the non-recursive random container generators.
simpleContainerPool=[randSimpleTuple,randSimpleList,randSimpleDict]

#Pool comprising non-recursive random container generators and non-container type generators.
completeSimplePool = simpleTypePool + simpleContainerPool     

#The definition for recursiveContainerPool is given after randDict.
#It must be declared here so that randomly generated recursive containers may refer
#to each other.
recursiveContainerPool=None

#Returns a recursive tuple and its Scala equivalent whose elements are generated randomly from each of the pools.
def randTuple(depth=1):
    pyList = []
    length = randint(1,MAX_TUPLE_LENGTH)
    if depth >= MAX_DEPTH:
        for i in range(length):
            pyList.append(choice(simpleTypePool)()[0])
    else:
        for i in range(length):
            if random() < RECURSE_RATE:
                pyList.append(choice(recursiveContainerPool)(depth+1)[0])
            else:
                pyList.append(choice(completeSimplePool)()[0])
    tup = tuple(pyList)
    return (tup,toScala(tup))

#Returns a recursive list and its Scala equivalent whose elements are generated randomly from each of the pools.
#The generated list has length at most MAX_LIST_LENGTH.
def randList(depth=1):
    pyList = []
    length = randint(0,MAX_LIST_LENGTH)
    if depth >= MAX_DEPTH:
        for i in range(length):
            pyList.append(choice(completeSimplePool)()[0])
    else:
        for i in range(length):
            if random() < RECURSE_RATE:
                pyList.append(choice(recursiveContainerPool)(depth+1)[0])
            else:
                pyList.append(choice(completeSimplePool)()[0])
    return (pyList,toScala(pyList))

#Returns a recursive dict and its Scala equivalent whose values are generated from each of the pools but whose
#keys are restricted to be from simpleTypePool.
#The reason for this is because mutable data is not hashable.
#While Python tuples are not mutable, Scala TupleNs ARE, so tuples are avoided as well.
#The generated list has length at most MAX_LIST_LENGTH.
def randDict(depth=1):
    pyDict = {}
    length = randint(0,MAX_DICT_LENGTH)
    if depth >= MAX_DEPTH:
        for i in range(length):
            pyDict[choice(simpleTypePool)()[0]] = choice(simpleTypePool)()[0]
    else:
        key=None
        value=None
        for i in range(length):
            key = choice(simpleTypePool)()[0]
            if random() < RECURSE_RATE:
                value = choice(recursiveContainerPool)(depth+1)[0]
            else:
                value = choice(completeSimplePool)()[0]
            pyDict[key]=value
    return (pyDict,toScala(pyDict))

#Pool comprising generators for containers which may have other containers as elements.
recursiveContainerPool=[randTuple,randList,randDict]
