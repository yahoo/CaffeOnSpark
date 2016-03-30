'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.

This module contains various routines involving Java's class objects for convenience.
'''

import string

'''
There is a distinction between py4j's "Java Classes" and Java's "Class Objects".
py4j's "Java Classes" are what is returned by finding the Java class in the jvm,
i.e., jvm.java.lang.Object.
In contrast, Java's "Class Objects" are what is returned by calling getClass() on an
instance, i.e., jvm.java.lang.Object().getClass() or java.lang.Class.forName(<Class name>)
Both types are frequently used: The former makes arrays of that type,allows
new instances of the class to be made, and static methods of that class to be called,
while the latter allow for the use of reflection.
'''

'''
Returns the Java class object for the class of the given name.
The name may not be abbreviated by stripping its package name.
'''
def javaClassObject(name):
    return jvm.java.lang.Class.forName(name)

'''
Returns the Java class object corresponding to the given Scala type.
'''
def javaClassFromScala(scalaType,mirror):
    return mirror.runtimeClass(scalaType)

'''
Returns the class object for java.lang.Object.
'''
def javaObjectClass():
    return javaClassObject("java.lang.Object")

'''
Returns the class object for java.lang.Class.
'''
def javaClassClass():
    return javaClassObject("java.lang.Class")

'''
Returns an empty array of Object.
'''
def emptyArray():
    return gateway.new_array(jvm.java.lang.Object,0)

'''
Returns an unassigned Object array with length num.
'''
def objectArray(num):
    return gateway.new_array(jvm.java.lang.Object,num)
    
'''
Returns an array of java class objects from a list of strs
of the full names of the classes.
'''
def classObjectArray(classNameList):
    arr = gateway.new_array(jvm.java.lang.Class,len(classNameList))
    for i,name in enumerate(classNameList):
        arr[i]=javaClassObject(name)
    return arr

'''
Returns a class object array in which each element is assigned to
the class for java.lang.Object.
The array has length num.
'''
def objectClassArray(num=1):
    arr = gateway.new_array(jvm.java.lang.Class,num)
    for i in range(num):
        arr[i] = javaObjectClass()
    return arr

'''
Returns True if the argument is a Java class object and False otherwise.
'''
def isClass(obj):
    return javaClassClass().isAssignableFrom(obj.getClass())

'''
Returns True if the given Java class object represents
an instantiable class and False otherwise.
'''
def isInstantiable(javaClass):
    return not (jvm.java.lang.reflect.Modifier.isAbstract(javaClass.getModifiers()) or javaClass.isInterface())

'''
Returns True if the given Java type represents
a parameterized type and False otherwise.
'''
def isParameterizedType(javaType):
    return javaClassObject("java.lang.reflect.ParameterizedType").isAssignableFrom(javaType.getClass())

'''
Returns True if the given Java type represents
a type variable and False otherwise.
'''
def isTypeVariable(javaType):
    return javaClassObject("java.lang.reflect.TypeVariable").isAssignableFrom(javaType.getClass())

'''
Returns True if the given Java type represents
a generic array type and False otherwise.
'''
def isGenericArrayType(javaType):
    return javaClassObject("java.lang.reflect.GenericArrayType").isAssignableFrom(javaType.getClass())

'''
Returns True if the given Java method takes a variable number of arguments and
False otherwise.
'''
def isVarArgs(method):
    try:
        return method.isVarArgs()
    except:
        return False

'''
Returns the Scala singleton Object for the class
named className.
'''
def getScalaSingleton(className):
    uni=jvm.scala.reflect.runtime.package.universe()
    rtm=uni.runtimeMirror(javaClassObject(className+'$').getClassLoader())
    clsm=rtm.classSymbol(javaClassObject(className+'$'))
    modSym=clsm.module()
    modMir=rtm.reflectModule(modSym)
    return modMir.instance()

'''
Returns the class tag of the given class.
'''
def getClassTag(javaClass):
    return getScalaSingleton("scala.reflect.ClassTag").apply(javaClass)

'''
Returns a Scala List of the parameter types of a Scala method.
'''
def getParameterTypes(scalaMethodSymbol):
    retList=[]
    it1=scalaMethodSymbol.paramss().toIterator()
    while it1.hasNext():
        it2=it1.next().toIterator()
        while it2.hasNext():
            signature=it2.next().typeSignature()
            if str(signature)[-1]=='*':
                retList.append(getGenericParameterTypes(signature).head())
            else:
                retList.append(signature)
    return retList

'''
Returns a Scala List of the generic parameter types of a Scala type.
'''
def getGenericParameterTypes(scalaType):
    return jvm.com.yahoo.ml.caffe.python.General.getTypeParameters(scalaType)

'''
Returns the global Scala ExecutionContext.
'''
def getScalaExecutionContext():
    className="scala.concurrent.ExecutionContext"
    return javaMethodObject(className+'$',"global",classObjectArray([])).invoke(getScalaSingleton(className),emptyArray())

'''
Returns the Java method object of name methodName with the class named className taking arguments
specified by a list of the names of the argument classes.
'''            
def javaMethodObject(className,methodName,argumentClassNameList=["java.lang.Object"]):
    return javaClassObject(className).getMethod(methodName,classObjectArray(argumentClassNameList))    

'''
Returns a Java method of name methodName of the class className
which has no arguments.
'''
def arglessJavaMethodObject(className,methodName):
    return javaMethodObject(className,methodName,[])

'''
Returns the Py4J JavaClass object equivalent to the Java class
object javaClass.
'''
def javaClassToPy4JClass(javaClass):
    return jvm.__getattr__(javaClass.getName())

'''
Returns true is classObj most-likely represents a Scala Tuple class.
Due to some Scala magic, it is difficult to know with absolute certainty
without using unwieldy code, though this method should be correct in all cases
that are not constructs used to contradict it.
'''
def isTupleClass(classObj):
    name = classObj.getName()
    dollarSign=name.find('$')
    if dollarSign != -1:
        name=name[0:name.find('$')]
    if len(name) > 11 and name[0:11]=="scala.Tuple":
        if len(name) == 12 and name[-1] in string.digits:
            return True
        if len(name) == 13:
            return (name[-2]=='1' and name[-1] in string.digits) or (name[-2]=='2' and name[-1] in string.digits and int(name[-1]) in range(3))
    return False
