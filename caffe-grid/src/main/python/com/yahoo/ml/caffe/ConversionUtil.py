'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.

This module contains methods for converting from Python types to their
functionally equivalent Scala and Java types and vice-versa.
Namely, it contains the functions toJava,toScala, toPython, and wrapClass.
Elements within containers are also converted.
The currently supported conversion types are:

===PYTHON========SCALA/JAVA=======
int               <===>Int/int
        
long              <===>Long/long

float             <===>Double/double

bool              <===>Boolean/boolean

None              <===>null/null

unicode str       <===>String/String

byte str          ====>String/String 
        
list              ====>scala.collection.immutable.Vector/java.util.ArrayList
                  <====scala.collection.Iterable/java.lang.Iterable
                  <====Array/[]

dict              ====>scala.collection.mutable.HashMap/java.util.HashMap
                  <====scala.collection.Map/java.util.Map

set               ====>scala.collection.mutable.HashSet/java.util.HashSet
                  <====scala.collection.Set/java.util.Set
       
tuple             <===>Scala:TupleN where N=len(tuple)
0<length<23       ====>Java:java.util.ArrayList

tuple             ====>scala.collection.immutable.Vector/java.util.ArrayList
length=0,>23

rdd               <===>org.apache.spark.rdd.RDD/org.apache.spark.api.java.JavaRDD
Note: Supported conversion types for RDDs are limited.
See com.yahoo.ml.caffe.python.RDDConversions.scala for details.

SparkContext      ===>Scala SparkContext/Java SparkContext.
             
SQLContext        ===>Scala SQLContext/Java SQLContext.

Future result     <====scala.concurrent.Future

bytearray         <===>Array[Byte]/byte[]
                  <====java.nio.ByteBuffer
'''

from py4j.protocol import Py4JJavaError
from ReflectionUtil import *
from GeneralUtil import isPrefix,unCamelCase,getTrailingNumber
import __builtin__
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark import rdd,SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.common import _java2py,_py2java
from collections import deque
import traceback

#The maximum time to wait for Scala Futures to complete.
awaitSeconds=600

_DEBUG=False

#Basically forward declarations.
_typeToCommonClass = None
_typeToPy4JScalaClass = None
_typeToPy4JJavaClass = None
_pythonToScalaConversions = None
_pythonToJavaConversions = None
_javaToPythonConversions = None
_expectedParameterizedConversions = None
_expectedRawConversions = None
_registeredParameterizedConversions = {}
_registeredRawConversions = {}
_wrappedClasses = {}
_objectMethodNames=frozenset(["$asInstanceOf","$isInstanceOf","synchronized","$hash$hash","$eq$eq","ne","eq","notifyAll",
                              "notify","clone","hashCode","toString","equals","wait","finalize","asInstanceOf","isInstanceOf",
                              "$bang$bang","$eq$eq"])

def _initTypeToCommonClass():
    #Initializes the global variable _typeToCommonClass if it is not initialized already.
    #An action such as this is done because _typeToCommonClass refers to the jvm on module level,
    #which can sometimes be problematic.
    global _typeToCommonClass
    #Type to Class conversions which are the same for both Java and Scala.
    if _typeToCommonClass == None:
        _typeToCommonClass={
            bool    : lambda x : jvm.java.lang.Boolean.TYPE,
            int     : lambda x : jvm.java.lang.Integer.TYPE,
            long    : lambda x : jvm.java.lang.Long.TYPE,
            float   : lambda x : jvm.java.lang.Double.TYPE,
            str     : lambda x : javaClassObject("java.lang.String"),
            unicode : lambda x : javaClassObject("java.lang.String")}

def _initTypeToPy4JScalaClass():
    global _typeToPy4JScalaClass
    _initTypeToCommonClass()
    #Type to Class conversions which are unique to Scala.
    if _typeToPy4JScalaClass == None:
        _typeToPy4JScalaClass={
            list  : lambda x : javaClassObject("scala.collection.immutable.Vector"),
            set   : lambda x : javaClassObject("scala.collection.mutable.HashSet"),
            dict  : lambda x : javaClassObject("scala.collection.mutable.HashMap"),
        tuple : _py4jTupleClass}
        _typeToPy4JScalaClass.update(_typeToCommonClass)

def _initTypeToPy4JJavaClass():
    global _typeToPy4JJavaClass
    _initTypeToCommonClass()
    #Type to Class conversion which are unique to Java.
    if _typeToPy4JJavaClass == None:
        _typeToPy4JJavaClass={
            list  : lambda x : javaClassObject("java.util.ArrayList"),
            set   : lambda x : javaClassObject("java.util.HashSet"),
            dict  : lambda x : javaClassObject("Java.util.HashMap"),
            tuple : lambda x : javaClassObject("java.util.ArrayList")}
        _typeToPy4JJavaClass.update(_typeToCommonClass)

def _initPythonToScalaConversions():
    global _pythonToScalaConversions
    if _pythonToScalaConversions == None:
        #Maps a Python type to a conversion routine for that type.
        _pythonToScalaConversions = {
            list         : toScalaVector,
            dict         : toScalaHashMap,
            tuple        : toScalaTuple,
            set          : toScalaHashSet,
            SQLContext   : toScalaSQLC,
            SparkContext : toScalaSC}

def _initPythonToJavaConversions():
    global _pythonToJavaConversions
    if _pythonToJavaConversions == None:
        #Dictionary from a Python type to a converter for the corrresponding Scala type.
        _pythonToJavaConversions = {
            list         : toJavaArrayList,
            dict         : toJavaHashMap,
            tuple        : toJavaArray,
            set          : toJavaHashSet,
            SQLContext   : toJavaSQLC,
            SparkContext : toJavaSC}
        
def _initJavaToPythonConversions():
    global _javaToPythonConversions
    if _javaToPythonConversions == None:
        #A list of tuples where the entry is a supported class and the value is the conversion method.
        _javaToPythonConversions=[
            (javaClassObject("org.apache.spark.rdd.RDD"),          fromScalaRDD),
            (javaClassObject("org.apache.spark.api.java.JavaRDD"), fromJavaRDD),
            (javaClassObject("scala.collection.Map"),              fromScalaMap),
            (javaClassObject("java.util.Map"),                     fromJavaMap),
            (javaClassObject("scala.collection.Set"),              fromScalaSet),
            (javaClassObject("java.util.Set"),                     fromJavaSet),
            (javaClassObject("scala.collection.Iterable"),         fromScalaIterable),
            (javaClassObject("java.lang.Iterable"),                fromJavaIterable),
            (javaClassObject("scala.concurrent.Future"),           fromScalaFuture),
            (javaClassObject("java.nio.ByteBuffer"),               fromJavaByteBuffer)]

def _initExpectedParameterizedConversion():
    global _expectedParameterizedConversions
    if _expectedParameterizedConversions == None:
        #Maps Java parameterized types to their Python conversion routines.
        _expectedParameterizedConversions=[
            (javaClassObject("scala.collection.mutable.Builder"), applyExpectedBuilderConversion),
            (javaClassObject("scala.collection.Map"),             applyExpectedScalaMapConversion),
            (javaClassObject("scala.collection.Traversable"),     applyExpectedTraversableConversion),
            (javaClassObject("java.util.Map"),                    applyExpectedJavaMapConversion),
            (javaClassObject("java.util.Collection"),             applyExpectedCollectionConversion),
            (javaClassObject("scala.concurrent.Future"),          applyExpectedFutureConversion)]

def _initExpectedRawConversions():
    global _expectedRawConversions
    if _expectedRawConversions == None:
        #Maps Java class objects to their Python conversion routines.
        _expectedRawConversions=[
            (javaClassObject("java.nio.ByteBuffer"),              applyExpectedByteBufferConversion),
            (javaClassObject("java.lang.Long"),                   long),
            (jvm.java.lang.Long.TYPE,                             long),
            (javaClassObject("scala.Long"),                       long),
            (javaClassObject("java.lang.Integer"),                int),
            (jvm.java.lang.Integer.TYPE,                          int),
            (javaClassObject("scala.Int"),                        int)]


'''
Gets the Scala Tuple class corresponding to the given Python tuple,
based on its length.
If its length is 0 or larger than 22, a Scala Vector class object is returned instead.
'''
def _py4jTupleClass(pyTuple):
    if len(pyTuple) > 0 and len(pyTuple) < 23:
        return javaClassObject("scala.Tuple"+str(len(pyTuple)))
    #In Scala array conversions, tuples which are empty or too long will be converted to Scala Vectors.
    return javaClassObject("scala.collection.immutable.Vector")
               
'''
Maps a Python object to a Scala class Object
corresponding to its type.
'''
def toScalaClass(obj):
    _initTypeToPy4JScalaClass()
    try:
        return _typeToPy4JScalaClass[type(obj)](obj)
    except:
        return javaObjectClass()

'''
Maps a Python object to the Java class Object
corresponding to its type.
'''
def toJavaClass(obj):
    _initTypeToPy4JJavaClass()
    try:
        return _typeToPy4JJavaClass[type(obj)](obj)
    except:
        return javaObjectClass()

'''
Helper method for converting Python iterables to Java/Scala arrays.
'''
def _toArrayHelp(pyIterable,classConverter,objectConverter):
    if len(pyIterable) == 0:
        return emptyArray()
    isConsistent=True
    componentClass = classConverter(pyIterable[0])
    #We can start at 1 because the class name is determined from the zeroth element.
    i=1
    while i < len(pyIterable) and isConsistent:
        if not classConverter(pyIterable[i]) == componentClass:
            #The array has inconsistent type and needs to be casted to an Object array instead.
            isConsistent=False
        i+=1
    array=None
    if isConsistent:
        array=gateway.new_array(javaClassToPy4JClass(componentClass),len(pyIterable))
    else:
        array=objectArray(len(pyIterable))
    for i,obj in enumerate(pyIterable):
        array[i]=objectConverter(obj)
    return array

'''
Returns an array of Scala objects corresponding to the elements of pyIterable.
If pyIterable contains only one type and that type has a corresponding Scala type/class, the pyIterable
is implicitly converted to an array of that type.
Otherwise, pyIterable is converted into an array of Objects.
'''
def toScalaArray(pyIterable):
    return _toArrayHelp(pyIterable,toScalaClass,toScala)

'''
Returns an array of Java objects corresponding to the elements of pyIterable.
If pyIterable contains only one type and that type has a corresponding Java type/class, the pyIterable
is implicitly converted to an array of that type.
Otherwise, pyIterable is converted into an array of Objects.
'''
def toJavaArray(pyIterable,dim=1):
    return _toArrayHelp(pyIterable,toJavaClass,toJava)

'''
Returns an array type java.lang.Object with the elements of the given iterable object.
Does not perform any conversions.
'''
def toObjectArray(pyIterable):
    if len(pyIterable)==0:
        return emptyArray()
    array=objectArray(len(pyIterable))
    for i,obj in enumerate(pyIterable):
        array[i]=obj
    return array

'''
Converts some iterable Python object to a Scala Vector.
Vector was chosen to be the default list conversion target for Scala because it is Scala's
primary random-access container.
'''
def toScalaVector(pyIterable):
    appendMethod = jvm.java.lang.Class.forName("scala.collection.immutable.VectorBuilder").getMethod("$plus$eq",objectClassArray(1))
    vectorBuilder = jvm.scala.collection.immutable.VectorBuilder()
    parArray = gateway.new_array(jvm.java.lang.Object,1)
    for i in pyIterable:
        parArray[0] = toScala(i)
        appendMethod.invoke(vectorBuilder,parArray)
    return vectorBuilder.result()

'''
Converts some iterable Python object to a Java ArrayList.
ArrayList was chosen to be the default list conversion target for Java because it is Java's
primary random-access container.
'''
def toJavaArrayList(pyIterable):
    arrList = jvm.java.util.ArrayList()
    for i in pyIterable:
        arrList.add(toJava(i))
    return arrList

'''
Converts some iterable Python object to a scala.TupleN whenever 0<len(pyIterable)<23.
Otherwise, pyIterable is converted to an array.
'''
def toScalaTuple(pyIterable):
    if len(pyIterable)==0 or len(pyIterable)>22:
        return toScalaArray(pyIterable)
    return jvm.__getattr__("scala.Tuple"+str(len(pyIterable)))(*tuple(map(toScala,pyIterable))) 

'''
Converts some iterable Python object to a Scala mutable HashSet.
HashSet was chosen to be the default set conversion target for Scala
because it is the most similar to a Python set, which also hashes.
'''
def toScalaHashSet(pyIterable):
    scalaSet = jvm.scala.collection.mutable.HashSet()
    for i in pyIterable:
        scalaSet.add(toScala(i))

'''
Converts some iterable Python object to a Java HashSet.
HashSet was chosen to be the default set conversion target for Java 
because it is the most similar to a Python set,which also hashes.
'''
def toJavaHashSet(pyIterable):
    javaSet = jvm.java.util.HashSet()
    for i in pyIterable:
        javaSet.add(toJava(i))
    return javaSet

'''
Converts some iterable Python object to a Scala mutable HashMap.
HashMap was chosen to be the default dict conversion target for Scala
because it is the most similar to a Python dict, which also hashes.
'''
def toScalaHashMap(pyDict):
    hashMap = jvm.scala.collection.mutable.HashMap()
    for k,v in pyDict.iteritems():
        hashMap.put(toScala(k),toScala(v))
    return hashMap

'''
Converts some iterable Python object to a Java HashMap.
HashMap was chosen to be the default dict conversion target for Java 
because it is the most similar to a Python dict,which also hashes.
'''
def toJavaHashMap(pyDict):
    hashMap = jvm.java.util.HashMap()
    for k,v in pyDict.iteritems():
        hashMap.put(toJava(k),toJava(v))
    return hashMap

'''
Converts a Python RDD to a Scala RDD.
Succeeds as long as each type in the RDD is supported by the conversion dictionaries.
Additional types are also supported : see RDDConversions.scala.    
'''
def toScalaRDD(pyRDD):        
    pyRDD=pyRDD._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return jvm.com.yahoo.ml.caffe.python.RDDConverter.pythonToScala(pyRDD._jrdd.rdd())

'''
Converts a Python RDD to a Java RDD.
This has already been done and the documentation for _py2java 
may be found at https://github.com/apache/spark/blob/master/python/pyspark/mllib/common.py
'''
def toJavaRDD(pyRDD):
    return _py2java(pyRDD)

'''
Converts a Python SparkContext to a Scala SparkContext.
'''
def toScalaSC(pySc):
    return pySc._jsc.sc()

'''
Converts a Python SparkContext to a Java SparkContext.
'''
def toJavaSC(pySc):
    return pySc._jsc

'''
Converts a Python SQLContext to a Scala SparkContext.
'''
def toScalaSQLC(pySQLc):
    return jvm.org.apache.spark.sql.SQLContext(pySQLc._jsc.sc())

'''
Converts a Python SQLContext to a Java SparkContext.
'''
def toJavaSQLC(pySQLc):
    return pySQLc._scala_SQLContext

'''
Converts a Java Array to a Python List.
Every element in the array is converted to its Python
equivalent if possible.
'''
def fromArray(array):
    pyList = []
    for i in array:
        pyList.append(toPython(i))
    return pyList

'''
Converts a Scala Tuple to a Python Tuple.
Every element in the tuple is converted to its Python
equivalent if possible.
'''
def fromScalaTuple(scalaTuple):
    elements=[]
    for i in range(1,scalaTuple.productArity()+1):
        elements.append(toPython(scalaTuple.__getattr__("_"+str(i))())) #Reflection hack!
    return tuple(elements)

'''
Converts some scala.collection.Map into a Python Dict.
Every element in the map is converted to its Python
equivalent if possible.
'''
def fromScalaMap(scalaMap):
    pyDict = {}
    it = scalaMap.toIterator()
    while it.hasNext():
        element = it.next()
        pyDict[toPython(element._1())]=toPython(element._2())
    return pyDict

'''
Converts some java.util.Map into a Python Dict.
Every element in the map is converted to its Python
equivalent if possible.
'''
def fromJavaMap(javaMap):
    pyDict = {}
    it = javaMap.keySet().iterator()
    while it.hasNext():
        element = it.next()
        pyDict[toPython(element)]=toPython(javaMap.get(element))
    return pyDict

'''
Converts some scala.collection.Set into a Python set.
Every element in the set is converted to its Python
equivalent if possible.
'''
def fromScalaSet(scalaSet):

    pySet = set()
    it = scalaSet.toIterator()
    while it.hasNext():
        pySet.add(toPython(it.next()))
    return pySet
'''
Converts some java.util.Set into a Python set.
Every element in the set is converted to its Python
equivalent if possible.
'''
def fromJavaSet(javaSet):
    pySet = set()
    it = javaSet.iterator()
    while it.hasNext():
        pySet.add(toPython(it.next()))
    return pySet

'''
Converts some scala.collection.Iterable into a Python list.
Every element in the iterable is converted to its Python
equivalent if possible.
'''
def fromScalaIterable(scalaIterable):
    pyList = []
    it = scalaIterable.toIterator()
    while it.hasNext():
        pyList.append(toPython(it.next()))
    return pyList

'''
Converts some java.lang.Iterable into a Python list.
Every element in the iterable is converted to its Python
equivalent if possible.
'''
def fromJavaIterable(javaIterable):
    pyList = []
    it = javaIterable.iterator()
    while it.hasNext():
        pyList.append(toPython(it.next()))
    return pyList
'''
Converts a Scala RDD to a Python RDD.
See the Scala source file com.yahoo.ml.caffe.python.RDDConversions.scala for more information.    
'''
def fromScalaRDD(scalaRDD):
    return rdd.RDD(jvm.com.yahoo.ml.caffe.python.RDDConverter.scalaToPython(scalaRDD).toJavaRDD(),sc)

'''
Converts a Java RDD to a Python RDD.
This simply invokes _java2py, the documentation for which
may be found at https://github.com/apache/spark/blob/master/python/pyspark/mllib/common.py
'''
def fromJavaRDD(javaRDD):
    return _java2py(javaRDD)

'''
Converts a Future in Scala to to Python by returning its result.
The maximum amount of time that such results may be waited on is dictated by the global
variable awaitSeconds.
'''
def fromScalaFuture(scalaFuture):
    return toPython(jvm.scala.concurrent.Await.result(scalaFuture,jvm.scala.concurrent.duration.Duration.apply(str(awaitSeconds)+'s')))

'''
Converts a Java.nio.ByteBuffer to a Python bytearray..
'''
def fromJavaByteBuffer(byteBuffer):
    return byteBuffer.array()

'''
Returns a rough Scala equivalent of the given Python Object.
If no conversion is available, the original object is returned.
'''
def toScala(obj):
    _initPythonToScalaConversions()
    if type(obj) in _pythonToScalaConversions:
        return _pythonToScalaConversions[type(obj)](obj)
    elif isinstance(obj.__class__,JavaClassWrapper):
        return obj.javaInstance
    elif isinstance(obj,rdd.RDD):
        return toScalaRDD(obj)
    return obj
        
'''
Returns a rough Java equivalent of the given Python object.
If no conversion is available, the original object is returned.
'''
def toJava(obj):
    _initPythonToJavaConversions()
    if type(obj) in _pythonToJavaConversions:
        return _pythonToJavaConversions[type(obj)](obj)
    elif isinstance(obj.__class__,JavaClassWrapper):
        return obj.javaInstance
    elif isinstance(obj,rdd.RDD):
        return toJavaRDD(obj)
    return obj
        
'''
Returns a rough Python equivalent of the given Scala/Java object.
If no conversion is available, the original object is returned.
'''
def toPython(obj):
    _initJavaToPythonConversions()
    if hasattr(obj,"getClass"):
        if isTupleClass(obj.getClass()):
            return fromScalaTuple(obj)
        if obj.getClass().isArray():
            return fromArray(obj)
        if obj.getClass() in _wrappedClasses:
            return _wrappedClasses[obj.getClass()](obj)
        name=obj.getClass().getName()
        if name == "java.lang.Void" or name == "void" or name == "scala.Unit" or name == "scala.runtime.BoxedUnit":
            return
        for i in _javaToPythonConversions:
            if i[0].isInstance(obj):
                return i[1](obj)
    return obj


#WRAPPING

'''
Raised after a failure to successfully apply a conversion scheme
inferred from the argument types of a Java method.
'''
class ConversionSchemeException(Exception):
    '''
    Initializes a ConversionSchemeException with an error message.
    '''
    def __init__(self,msg):
        self.msg=msg
    '''
    Returns the str representation of the exception message.
    '''
    def __str__(self):
        return str(self.msg)


'''
Helper method used to get an Object array of converted
Python arguments to a Java method.
'''
def _getConvertedTuple(args,sym,defaults,mirror):
    expectedTypes=getParameterTypes(sym)
    if sym.isVarargs():
        argDifference=len(args)-len(expectedTypes)+1 #Makes things quite a bit easier.
        if sym.isJava():
            varArgs=objectArray(argDifference)
            for i in range(argDifference):
                varArgs[i]=applyExpectedConversion(args[-argDifference+i],expectedTypes[-1],mirror)
        else:
            varBuilder=getScalaSingleton("scala.collection.immutable.List").newBuilder()
            for i in range(argDifference):
                varBuilder.__getattr__("$plus$eq")(applyExpectedConversion(args[-argDifference+i],expectedTypes[-1],mirror))
            varArgs=varBuilder.result()
        newArgs=[]
        for i in range(len(expectedTypes)-1):
            newArgs.append(applyExpectedConversion(args[i],expectedTypes[i],mirror))
        newArgs.append(varArgs)
        return tuple(newArgs)
    else:
        argDifference=len(expectedTypes)-len(args)
        convertedList=[]
        for n,i in enumerate(args):
            convertedList.append(applyExpectedConversion(i,expectedTypes[n],mirror))
        #Substitute default arguments for absent arguments.
        for i in range(argDifference):
            convertedList.append(defaults[-argDifference+i])
        return tuple(convertedList)

'''
Attempts to call a Java method on Python arguments by reflectively
analyzing the parameter types of the method and applying a conversion scheme 
accordingly.
'''
def callJavaMethod(sym,javaInstance,defaults,mirror,*args):
    try:
        name=str(sym.name())
        if "<init>" in name:
            #Constructors are handled specially.
            return javaInstance(*_getConvertedTuple(args,sym,defaults,mirror))
        else:
            return toPython(javaInstance.__getattr__(name)(*_getConvertedTuple(args,sym,defaults,mirror)))
    #It is good for debugging to know whether the argument conversion was successful.
    #If it was, a Py4JJavaError may be raised from the Java code.
    #Otherwise, we raise a ConversionSchemeException.
    except Py4JJavaError as e:
        if _DEBUG:
            traceback.print_exc()
        if not javaClassObject("java.lang.IllegalArgumentException").isInstance(e.java_exception): 
            raise
    except:
        if _DEBUG:
            traceback.print_exc()
        raise ConversionSchemeException("No parameter conversion scheme exists for the sym "+str(sym.name)+" with arguments "+str(args)+"!")




'''
Converts an iterable Python object to an array by using a conversion routine based on the expected class.
'''
def applyExpectedArrayConversion(pyIterable,arrayType,mirror):
    componentType=getGenericParameterTypes(arrayType).head()
    fullName=javaClassFromScala(componentType,mirror).getName()
    array = gateway.new_array(jvm.__getattr__(fullName),len(pyIterable))
    for i in range(len(pyIterable)):
        array[i]=applyExpectedConversion(pyIterable[i],componentType,mirror)
    return array

'''
Converts an iterable Python object into a Scala Tuple using a conversion routine based on the expected class.
'''
def applyExpectedTupleConversion(pyIterable,tupleType,mirror):
    typeParameters=getGenericParameterTypes(tupleType)
    tupleClass=javaClassFromScala(tupleType,mirror)
    convertedList = []
    it = typeParameters.toIterator()
    i=0
    while it.hasNext():
        convertedList.append(applyExpectedConversion(pyIterable[i],it.next(),mirror))
        i+=1
    return jvm.__getattr__(tupleClass.getClass().getName())(*tuple(convertedList))

'''
Converts an iterable Python object into a Scala builder using a conversion routine based on the expected class.
'''
def applyExpectedBuilderConversion(pyIterable,builderType,mirror):
    builderClass=javaClassFromScala(builderType,mirror).head()
    builder=builderClass.newInstance()
    typeParameter = getGenericParameterTypes(builderType).head()
    for i in pyIterable:
        builder.__getattr__("$plus$eq")(applyExpectedConversion(i,typeParameter,mirror))
    return builder

'''
Converts a Python map object into a Scala Map using a conversion routine based on the expected class.
'''
def applyExpectedScalaMapConversion(pyDict,mapType,mirror):
    typeParameters=getGenericParameterTypes(mapType)
    mapClass = javaClassFromScala(mapType,mirror)
    builder=getScalaSingleton(mapClass.getName()).newBuilder()
    for i in pyDict:
        builder.__getattr__("$plus$eq")(jvm.scala.Tuple2(applyExpectedConversion(i,typeParameters.head(),mirror),applyExpectedConversion(pyDict[i],typeParameters.last(),mirror)))
    return builder.result()

'''
Converts a Python iterable ojbect into a Scala Traversable using a conversion routine based on the expected class.
'''
def applyExpectedTraversableConversion(pyIterable,traversableType,mirror):
    typeParameter=getGenericParameterTypes(traversableType).head()
    traversableClass=javaClassFromScala(traversableType,mirror)
    builder = getScalaSingleton(traversableClass.getName()).newBuilder()
    for i in pyIterable:
        builder.__getattr__("$plus$eq")(applyExpectedConversion(i,typeParameter,mirror))
    return builder.result()

'''
Converts a Python map object into a Java Map using a conversion routine based on the expected class.
'''
def applyExpectedJavaMapConversion(pyDict,mapType,mirror):
    typeParameters=getGenericParameterTypes(mapType)
    mapClass=javaClassFromScala(mapType,mirror)
    if isInstantiable(mapClass):
        javaMap = mapClass.newInstance()
    else:
        javaMap = jvm.java.util.HashMap()
    for k,v in pyDict.iteritems():
        map.put(applyExpectedConversion(k,typeParameters.head(),mirror),applyExpectedConversion(v,typeParameters.last(),mirror))
    return javaMap

'''
Converts a Python map object into a Java Collection using a conversion routine based on the expected class.
'''
def applyExpectedCollectionConversion(pyIterable,collectionType,mirror):
    collectionClass=javaClassFromScala(collectionType,mirror)
    if isInstantiable(collectionClass):
        javaCollection = collectionClass.newInstance()
    else:
        javaCollection = jvm.java.util.ArrayList()
    typeParameter=getGenericParameterTypes(collectionType).head()
    #This indirection is done to ensure the creation complexity is no more than O(n).
    linkedList = jvm.java.util.LinkedList()
    for i in pyIterable:
        linkedList.add(applyExpectedConversion(i,typeParameter,mirror))
    javaCollection.addAll(linkedList)
    return javaCollection

'''
Converts a Python Object to a scala.concurrent.Future using a conversion routine based on the expected class.
'''
def applyExpectedFutureConversion(pyObj,futureType,mirror):
    futureInst=getScalaSingleton("scala.concurrent.Future")
    convertedResult=applyExpectedConversion(pyObj,getGenericParameterTypes(futureType).head(),mirror)
    return futureInst.apply(jvm.scala.ref.SoftReference(convertedResult),getScalaExecutionContext())

'''
Converts a Python bytearray into a java.nio.ByteBuffer.
'''
def applyExpectedByteBufferConversion(byteArray):
    byteBuffer=jvm.java.nio.ByteBuffer.allocate(len(byteArray))
    for i in byteArray:
        byteBuffer.put(i)
    return byteBuffer

'''
Adds a conversion routine to the dict of expected parametrized conversions.
'''
def addExpectedParameterizedConversion(scalaType,function):
    _initExpectedParameterizedConversion()
    _expectedParameterizedConversions.append((scalaType,function))

'''
Adds a conversion routine to the dict of expected class object conversions.
'''    
def addExpectedRawConversion(javaClass,function):
    _initExpectedRawConversions()
    _expectedRawConversions.append((javaClass,function))


def applyExpectedConversion(pyObj,scalaType,mirror):
    if pyObj == None:
        return
    _initExpectedParameterizedConversion()
    _initExpectedRawConversions()
    sym=scalaType.typeSymbol().asType()
    if sym.isAbstractType() or sym.isExistential():
        return applyExpectedConversion(pyObj,scalaType.erasure(),mirror)
    else:
        javaClass=javaClassFromScala(scalaType,mirror)
        if javaClass in _registeredParameterizedConversions:
            return _registeredParameterizedConversions[javaClass](pyObj,scalaType,mirror)
        if javaClass in _registeredRawConversions:
            return _registeredRawConversions[javaClass](pyObj)
        if javaClass.isArray():
            return applyExpectedArrayConversion(pyObj,scalaType,mirror)
        if isTupleClass(javaClass):
            return applyExpectedTupleConversion(pyObj,scalaType,mirror)
        if javaClassObject("org.apache.spark.api.java.JavaRDD").isAssignableFrom(javaClass):
            return toJavaRDD(pyObj)
        for tup in _expectedParameterizedConversions:
            if tup[0].isAssignableFrom(javaClass):
                _registeredParameterizedConversions[javaClass]=tup[1]
                return tup[1](pyObj,scalaType,mirror)
        for tup in _expectedRawConversions:
            if tup[0].isAssignableFrom(javaClass):
                _registeredRawConversions[javaClass]=tup[1]
                return tup[1](pyObj)
    return toScala(pyObj)



def javaMap(rdd,methodWrapper):
    return toPython(jvm.com.yahoo.ml.caffe.python.reflectionMapMethod(toScala(rdd,methodWrapper.javaMethods[0])))

'''
Wraps a Java (or Scala) method so that arguments given in Python and the return value will automatically be converted.
'''
class JavaMethodWrapper(object):
    '''
    Creates a JavaMethodWrapper from a list of Java syms
    which represent every candidate of a overriden method.
    In addition, the javaInstance on which to invoke it along with
    the default arguments are supplied.
    '''
    def __init__(self,syms,javaInstance,defaults,mirror):
        self.syms=syms
        self.defaults=defaults
        self.mirror=mirror
        self.javaInstance=javaInstance
    
    '''
    Attempts to call one of javaMethods with the given arguments.
    These arguments are converted based on the parameter types of the syms.
    Raises a ConversionSchemeException if it is unsuccessful in calling it.
    '''
    def __call__(self,*args):
        for i in self.syms:
            try:
                return callJavaMethod(i,self.javaInstance,self._evalDefaults(),self.mirror,*args)
            except Py4JJavaError:
                raise
            except:
                pass
        raise ConversionSchemeException('No parameter conversion scheme exists for the method "'+str(self.syms[0].name())+'" with arguments '+str(args)+"!")

    def _evalDefaults(self):
        defaultList=[]
        for i in self.defaults:
            defaultList.append(self.javaInstance.__getattr__(i)())
        defaultList += self._getImplicits()
        return defaultList
            
    def _getImplicits(self):
        #_getImplicits will attempt to give a list of values to be used as the implicit parameters
        #for some Scala method.
        #The value that will be used it the conversion of some global Python type into the expected type.
        methodSymbol=self.syms[0]
        paramss=methodSymbol.paramss()
        if paramss.size() == 0:
            return []
        params=paramss.apply(paramss.size()-1)
        if params.size() == 0 or not params.apply(0).isImplicit():
            return []
        implicitList=[]
        it=params.toIterator()
        while it.hasNext():
            current=it.next()
            try:
                implicitList.append(toScala(getattr(__builtin__,str(current.name()))))
            except:
                try:
                    implicitList.append(toScala(getattr(globals(),str(current.name()))))
                except:
                    return []
        return implicitList
    
class JavaClassWrapper(type):
    '''
    JavaClassWrapper serves as a metaclass for classes that wrap a Java class.
    Public fields of the wrapped class are accessible through <instance>.<attribute> notation.
    This is also true for variables which follow conventional getter naming conventions or are
    an automatically generated Scala getter.
    If the variable also has a setter method prefixed with "set" or an automatically generated
    Scala setter, the variables may be set via <instance>.<attribute>=<value> as well.
    Attempts are made to convert all method and constructor arguments to their required
    types, and conversions of the return types to Python are also made.
    '''
    class _WrappedPublicField(object):
        #A handler for a wrapped public field in a class built from JavaClassWrapper.
        
        def __init__(self,name,fieldType,mirror):
            #Initializes a _WrappedPublicField using the name
            #of the field and the type of the field.
            self.name=name
            self.mirror=mirror
            self.fieldType=fieldType
            
        '''
        Gets the Python value of the public field.
        '''
        def get(self,javaInstance):
            return toPython(javaInstance.__getattr__(self.name))
        
        '''
        Sets the public field to a conversion of the given Python value.
        '''
        def set(self,javaInstance,value):
            javaValue=applyExpectedConversion(value,self.fieldType,self.mirror)
            javaInstance.__setattr__(self.name,javaValue)
    
    class _WrappedGettableField(object):
        #A handler for a wrapped gettable field in a class built from JavaClassWrapper
        def __init__(self,name,getterName,fieldType):
            #Initializes a _WrappedGettableField using the field name,
            #the name of its getter, and its type.
            self.name=name
            self.getterName=getterName
            self.fieldType=fieldType
            self.javaName=getterName
        
        def get(self,javaInstance):
            #Gets the Python value of the field.
            return toPython(javaInstance.__getattr__(self.getterName)())
        
        def set(self,javaInstance,value):
            #Gettable fields are assumed to not be settable, and thus any attempt to
            #set them raises an AttributeError.
            raise AttributeError('No obvious setter for "'+self.name+'".')
  
    class _WrappedSettableField(_WrappedGettableField):
        #A handler for a wrapped settable field in a class built from JavaClassWrapper.
        def __init__(self,name,getterName,setterName,fieldType,mirror):
            #Initializes a _WrappedSettableField using the field name,
            #the getter name, the setter name, and the field type.
            self.setterName=setterName
            self.mirror=mirror
            super(JavaClassWrapper._WrappedSettableField,self).__init__(name,getterName,fieldType)

        def set(self,javaInstance,value):
            #Sets the field to a conversion of the given Python value.
            javaValue=applyExpectedConversion(value,self.fieldType,self.mirror)
            javaInstance.__getattr__(self.setterName)(javaValue)
       
    class _WrappedMethod(object):
        #A handler for a wrapped method in a class built from JavaClassWrapper.
        def __init__(self,sym,mirror):
            #Initializes a _WrappedMethod using the Java
            #Method object.
            self.syms=[sym]
            self.mirror=mirror
            self.defaults=()
            
        def get(self,javaInstance):
            return JavaMethodWrapper(self.syms,javaInstance,self.defaults,self.mirror)
        
        def set(self,javaInstance,value):
            #Method setting for syms of classes created by JavaClassWrapper is not supported,
            #and set will raise an AttributeError.
            raise AttributeError('Cannot set a Java Method!')

        def add(self,method):
            #Adds a method to the list of Java syms of the same name.
            self.syms.append(method)
            
        def setDefaults(self,defaults):
            #Sets the values to be treated as default arguments for this method.
            self.defaults=defaults
    
    @staticmethod
    def defaultInit(self,*args):
        '''
        For a class constructed by JavaClassWrapper, defaultInit is what will take
        the place of __init__ for an instance of that class unless that class defines
        the method "customInit".
        defaultInit will first see if there is only 1 argument and that argument is an instance of
        the Java class for which JavaClassWrapper has wrapped.
        If this is the case, defaultInit simply sets the underlying Java instance to the argument.
        Otherwise, defaultInit will attempt to call a constructor for the class on the converted arguments.
        @param args Python arguments to convert to Java constructor arguments.
        '''
        if len(args) == 1 and type(args[0]) != type(self) and hasattr(args[0],"getClass") and args[0].getClass() == self.javaClass:
            object.__setattr__(self,"javaInstance",args[0])
        else:
            object.__setattr__(self,"javaInstance",self._constructors(*args))
            
    @staticmethod
    def _initMethod(self,*args):
        #_initMethod is what __init__ directly points to.
        #If customInit is defined for the class of the instance, 
        #customInit is called.
        #Otherwise, defaultInit is called.
        if "customInit" in self.__class__.__dict__:
            return self.customInit(*args)
        else:
            self.defaultInit(*args)
    
    @staticmethod
    def defaultGetAttr(self,name):
        '''
        defaultGetAttr is what will take the place of __getattr__ for
        an instance of a class constructed by JavaClassWrapper if
        customGetAttr is not defined.
        It attempts to find and convert a non-static member of 
        the underlying Java instance.
        Note that __getattr__ is only called if the class has not explicitly
        defined an attribute of the given name.
        @param name Name of the field to get.
        @return The field converted from Java to Python.
        '''
        try:
            return self._javaDict[name].get(self.javaInstance)
        except AttributeError:
            raise AttributeError(self.javaClass.getName()+' has no accessible member "'+name+'".')
        
    @staticmethod    
    def _getAttrMethod(self,name):
        #_getAttrMethod is what __getattr__ directly points to.
        #If customGetAttr is defined, it calls it.
        #Otherwise, it calls defaultGetAttr.
        if "customGetAttr" in self.__class__.__dict__:
            return self.customGetAttr(name)
        else:
            return self.defaultGetAttr(name)
    
    
    @staticmethod        
    def defaultSetAttr(self,name,value):
        '''
        defaultSetAttr is what will take the place of __setattr__ for an
        instance of a class constructed by JavaClassWrapper if customSetAttr
        is not defined.
        It attempts to find and set a non-static member of the underlying Java instance
        to a converted Python value.
        Note that __setattr__ is called even if the class has explicitly defined an attribute
        of the given name.
        @param name Name of the field to set.
        @param value Value to convert set the attribute to.
        '''
        if name in self.__dict__:
            object.__setattr__(self,name,value)
        else:
            try:
                self._javaDict[name].set(self.javaInstance,value)
            except AttributeError:
                raise AttributeError(self.javaClass.getName()+' has no settable member "'+name+'".')
    
    
    
    @staticmethod        
    def _setAttrMethod(self,name,value):
        #_setAttrMethod is what __setattr__ directly points to.
        #If customSetAttr is defined, it calls it.
        #Otherwise, it calls defaultSetAttr.
        if "customSetAttr" in self.__class__.__dict__:
            return self.customSetAttr(name,value)
        else:
            self.defaultSetAttr(name,value)
    
    @staticmethod
    def strMethod(self):
        '''
        strMethod replaces __str__ by calling toString on the underlying Java instance.
        '''
        return self.javaInstance.toString()    
    
    def __getattr__(self,name):
        '''
        Determines how the wrapped class (not instance) will access attributes.
        Giving a definition for this method allows for static class members to be implicitly accessable.
        @param name The name of the class attribute to get.
        '''
        try:
            return self._staticJavaDict[name].get(self._py4jClass)
        except:
            raise AttributeError(self.javaClass.getName()+' has no accessible static member "'+name+'".')
    
    def __setattr__(self,name,value):
        '''
        Determines how the wrapped class (not instance) will access attributes.
        Giving a definition for this method allows for static class members to be implicitly settable.
        @param name The name of the class attribute to set.
        @param value The value to set the attribute to.
        '''
        try:
            self._staticJavaDict[name].set(self._staticInstance)
        except:
            raise AttributeError(self.javaClass.getName()+' has no settable static member "'+name+'".')
       
    def __str__(self):
        '''
        Returns the string representation of the underlying Java class.
        ''' 
        return self.javaClass.getName()
    
    @staticmethod
    def _intDictToList(intDict):
        #Converts a dictionary which contains entries as integers from M to N for some M and N with M<=N.
        #Such dictionaries are created when elements are put in "out of order" with the size required unknown.
        dictList=[]
        #dictMin is the argument number of the first default argument.
        dictMin=None
        for i in intDict:
            dictList.append(0)
            if dictMin==None or i<dictMin:
                dictMin=i
        for k,v in intDict.iteritems():
            dictList[k-dictMin]=v
        return dictList
    
    @staticmethod
    def _processMembers(scalaType,mirror,javaDict,getterPrefixes,constructors=None):
        nonPublicFields={}
        methodDefaults={}
        it=scalaType.members().toIterator()
        lowestConstructorDefault=None
        highestConstructorDefault=None
        while it.hasNext():
            i=it.next()
            name = str(i.name())
            if name not in _objectMethodNames:
                if i.isMethod():
                    i=i.asMethod()
                    if constructors != None and i.isConstructor() and i.owner() == scalaType.typeSymbol():
                        constructors.append(i)
                    elif "$default$" in name:
                        if "$lessinit$greater" in name:
                            conDefaultIndex=getTrailingNumber(name)
                            if lowestConstructorDefault == None:
                                lowestConstructorDefault=conDefaultIndex
                                highestConstructorDefault=conDefaultIndex
                            elif conDefaultIndex < lowestConstructorDefault:
                                lowestConstructorDefault=conDefaultIndex
                            elif conDefaultIndex > highestConstructorDefault:
                                highestConstructorDefault=conDefaultIndex
                        else:
                            methodName = name[:name.find('$default$')]
                            if methodName in methodDefaults:
                                methodDefaults[methodName][getTrailingNumber(name)]=str(i.name())
                            else:
                                methodDefaults[methodName]={getTrailingNumber(name) : str(i.name())}
                    elif i.isGetter():
                        if name in nonPublicFields:
                            nonPublicFields[name][0]=name
                            nonPublicFields[name][2]=i.returnType()
                        else:
                            #Sometimes the getter is present but the variable is not.
                            nonPublicFields[name]=[name,None,i.returnType()]
                    elif i.isSetter():
                        fieldName=name[:-4]
                        if fieldName in nonPublicFields:
                            nonPublicFields[fieldName][1]=name
                        else:
                            nonPublicFields[fieldName]=[None,name,None]
                    elif name in nonPublicFields:
                        nonPublicFields[name][0]=name #Getter of the same name.
                    elif isPrefix("set",name) and len(name) > 3:
                            fieldName=unCamelCase(3,name)
                            if fieldName in nonPublicFields:
                                nonPublicFields[fieldName][1]=name
                            else:
                                nonPublicFields[fieldName]=[None,name,None]
                    else:
                        for j in getterPrefixes:     
                            if isPrefix(j,name) and len(name)>len(j):
                                fieldName=unCamelCase(len(j),name)
                                if fieldName not in javaDict:
                                    if fieldName in nonPublicFields:
                                        nonPublicFields[fieldName][0]=name
                                    else:
                                        nonPublicFields[fieldName]=[name,None,None]
                    if name not in javaDict:
                        javaDict[name]=JavaClassWrapper._WrappedMethod(i,mirror)
                    elif isinstance(javaDict[name],JavaClassWrapper._WrappedMethod):
                        javaDict[name].add(i)
                elif i.isTerm() and (i.isVar() or i.isVal()):
                    if name[-1] == " ":
                        name=name[:-1]
                    i=i.asTerm()
                    if i.isJava() and i.isPublic(): 
                        javaDict[name]=JavaClassWrapper._WrappedPublicField(name,i.typeSignature())
                    elif str(i.setter()) != "<none>":
                        javaDict[name]=JavaClassWrapper._WrappedSettableField(name,str(i.getter().name()),str(i.setter().name()),i.typeSignature(),mirror)
                    elif str(i.getter()) != "<none>":
                        javaDict[name]=JavaClassWrapper._WrappedGettableField(name,str(i.getter().name()),i.typeSignature())        
                    elif name in javaDict: #Method of the same name in javaDict
                        if name in nonPublicFields:
                            nonPublicFields[name][0]=name
                            nonPublicFields[name][2]=i.typeSignature()
                        else:
                            nonPublicFields[name]=[name,None,i.typeSignature()]
                    elif name in nonPublicFields:
                        nonPublicFields[name][2]=i.typeSignature()
                    else:
                        signature=i.typeSignature()
                        if str(signature)[-1] != '*':
                            #We'll cross the <repeated> class bridge when we get there...
                            nonPublicFields[name]=[None,None,signature]
        for k,v in nonPublicFields.iteritems():
            if v[2] != None:
                if v[1] != None:
                    javaDict[k]=JavaClassWrapper._WrappedSettableField(k,v[0],v[1],v[2],mirror)
                elif v[0] != None:
                    javaDict[k]=JavaClassWrapper._WrappedGettableField(k,v[0],v[2])      
        for k,v in methodDefaults.iteritems():
            javaDict[k].setDefaults(JavaClassWrapper._intDictToList(v))
        return (lowestConstructorDefault,highestConstructorDefault)
    
    def __new__(cls,name,bases,dct):
        #Creates a class which is a wrapper for a specified Java class.
        #The Java class is specified by giving the class name in a field called
        #javaClassName.
        
        global _wrappedClasses
        #The javaClassName attribute needs to be defined in all classes made from JavaClassWrapper
        javaClass=javaClassObject(dct["javaClassName"])
        
        #_javaDict keeps track of all non-static fields and public syms.
        _javaDict={}
        
        #_staticJavaDict keeps track of all fields and public syms that are static.
        #It is used so that static syms may be called and static fields may be accessed using the class name, not an instance.
        _staticJavaDict={}
        
        #Prefixes to consider when searching for likely getters/setters.
        #By default, "get" and "is" are supported, though more can be specified
        #with the _prefixes attribute.
        _getterPrefixes=["get","is"]
        
        constructors=[]
        
        _py4jClass=javaClassToPy4JClass(javaClass)
        
        #Additional getter prefixes may be specified in an attribute name _prefixes
        #corresponding to some iterable object.
        if "_prefixes" in dct:
            for i in dct["_prefixes"]:
                _getterPrefixes.append(i)
        
        #The following lines define various attributes for the wrapped class.
        dct["__init__"]=JavaClassWrapper._initMethod
        
        dct["defaultInit"]=JavaClassWrapper.defaultInit
        
        dct["javaClass"]=javaClass
        
        dct["_javaDict"]=_javaDict
        
        dct["_staticJavaDict"]=_staticJavaDict
        
        dct["_getterPrefixes"]=_getterPrefixes

        dct["__getattr__"]=JavaClassWrapper._getAttrMethod
        
        dct["defaultGetAttr"]=JavaClassWrapper.defaultGetAttr

        dct["__setattr__"]=JavaClassWrapper._setAttrMethod
        
        dct["defaultSetAttr"]=JavaClassWrapper.defaultSetAttr

        dct["__str__"]=JavaClassWrapper.strMethod

        dct["_py4jClass"]=_py4jClass
        
        #Scala mirror used to use reflection specify to Scala.
        mirror=jvm.scala.reflect.runtime.package.universe().runtimeMirror(javaClass.getClassLoader())
        
        #The Scala type of this class.
        scalaType = mirror.classSymbol(javaClass).toType()
        
        foundModule=False
        
        try:
            scalaModuleType = scalaType.typeSymbol().companionSymbol().asModule().moduleClass().asType().toType()
            foundModule=True
        except:
            pass
        try:
            dct["_staticInstance"]=getScalaSingleton(javaClass.getName())
        except:
            dct["_staticInstance"]=_py4jClass   
        
        JavaClassWrapper._processMembers(scalaType,mirror,_javaDict,_getterPrefixes,constructors)
        if foundModule:
            defaultIndices=JavaClassWrapper._processMembers(scalaModuleType,mirror,_staticJavaDict,_getterPrefixes)
        else:
            defaultIndices=(None,None)
        constructorDefaultNames=[]
        
        if defaultIndices[0] != None:
            for i in range(defaultIndices[0],defaultIndices[1]+1):
                constructorDefaultNames.append("$lessinit$greater$default$"+str(i))
        
        dct["_constructors"]=JavaMethodWrapper(constructors,_py4jClass,constructorDefaultNames,mirror)
        
        return super(JavaClassWrapper,cls).__new__(cls,name,bases,dct)
    
    def __init__(self,name,bases,dct):
        #So we may make a Python instance from a Java instance.
        _wrappedClasses[self.javaClass]=self
        super(JavaClassWrapper,self).__init__(name,bases,dct)

def wrapClass(className):
    """wrapClass creates a class from a Java class on demand using the full class name.
    
    Optionally, the name of the class may be given as the second argument.
    If it is not, the class's simple name is used (i.e., java.util.ArrayList=>ArrayList).
    """
    if javaClassObject(className) in _wrappedClasses:
        return _wrappedClasses[javaClassObject(className)]
    wrappedName=str(javaClassObject(className).getSimpleName())
    try:
        return JavaClassWrapper(wrappedName,(object,),{"javaClassName" : className})
    except:
        print "Wrapping of "+className+" failed!"
        raise
