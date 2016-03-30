/**
* Copyright 2016 Yahoo Inc.
* Licensed under the terms of the Apache 2.0 license.
* Please see LICENSE file in the project root for terms.
*/

package com.yahoo.ml.caffe.python

import scala.collection.immutable.List
import scala.collection.immutable.Vector
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.math.BigInt
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.apache.spark.rdd.RDD
import scala.math.ScalaNumericAnyConversions
import scala.language.implicitConversions
import scala.collection.Iterator
import scala.reflect.runtime.universe._
import scala.collection.mutable.Builder


//If you want a global variable, this is pretty much how you have to do it from what I know.

//Copied from cPickle documentations, weird ordering included.
case object MARK extends            OptionCode('(') // push special markobject on stack
case object STOP extends            OptionCode('.') // every pickle ends with STOP
case object POP extends             OptionCode('0') // discard topmost stack item
case object POP_MARK extends        OptionCode('1') // discard stack top through topmost markobject
case object DUP extends             OptionCode('2') // duplicate top stack item
case object FLOAT extends           OptionCode('F') // push float object; decimal string argument
case object INT extends             OptionCode('I') // push integer or bool; decimal string argument
case object BININT extends          OptionCode('J') // push four-byte signed int
case object BININT1 extends         OptionCode('K') // push 1-byte unsigned int
case object LONG extends            OptionCode('L') // push long; decimal string argument
case object BININT2 extends         OptionCode('M') // push 2-byte unsigned int
case object NONE extends            OptionCode('N') // push None
case object PERSID extends          OptionCode('P') // push persistent object; id is taken from string arg
case object BINPERSID extends       OptionCode('Q') //  "       "         "  ;  "  "   "     "  stack
case object REDUCE extends          OptionCode('R') // apply callable to argtuple, both on stack
case object STRING extends          OptionCode('S') // push string; NL-terminated string argument
case object BINSTRING extends       OptionCode('T') // push string; counted binary string argument
case object SHORT_BINSTRING extends OptionCode('U') //  "     "   ;    "      "       "      " < 256 bytes
case object UNICODE extends         OptionCode('V') // push Unicode string; raw-unicode-escaped'd argument
case object BINUNICODE extends      OptionCode('X') //   "     "       "  ; counted UTF-8 string argument
case object APPEND extends          OptionCode('a') // append stack top to list below it
case object BUILD extends           OptionCode('b') // call __setstate__ or __dict__.update()
case object GLOBAL extends          OptionCode('c') // push self.find_class(modname, name); 2 string args
case object DICT extends            OptionCode('d') // build a dict from stack items
case object EMPTY_DICT extends      OptionCode('}') // push empty dict
case object APPENDS extends         OptionCode('e') // extend list on stack by topmost stack slice
case object GET extends             OptionCode('g') // push item from memo on stack; index is string arg
case object BINGET extends          OptionCode('h') //   "    "    "    "   "   "  ;   "    " 1-byte arg
case object INST extends            OptionCode('i') // build & push class instance
case object LONG_BINGET extends     OptionCode('j') // push item from memo on stack; index is 4-byte arg
case object LIST extends            OptionCode('l') // build list from topmost stack items
case object EMPTY_LIST extends      OptionCode(']') // push empty list
case object OBJ extends             OptionCode('o') // build & push class instance
case object PUT extends             OptionCode('p') // store stack top in memo; index is string arg
case object BINPUT extends          OptionCode('q') //   "     "    "   "   " ;   "    " 1-byte arg
case object LONG_BINPUT extends     OptionCode('r') //   "     "    "   "   " ;   "    " 4-byte arg
case object SETITEM extends         OptionCode('s') // add key+value pair to dict
case object TUPLE extends           OptionCode('t') // build tuple from topmost stack items
case object EMPTY_TUPLE extends     OptionCode(')') // push empty tuple
case object SETITEMS extends        OptionCode('u') // modify dict by adding topmost key+value pairs
case object BINFLOAT extends        OptionCode('G') // push float; arg is 8-byte float encoding

case object PROTO extends           OptionCode(0x80.toByte) // identify pickle protocol
case object NEWOBJ extends          OptionCode(0x81.toByte) // build object by applying cls.__new__ to argtuple
case object EXT1 extends            OptionCode(0x82.toByte) // push object from extension registry; 1-byte index
case object EXT2 extends            OptionCode(0x83.toByte) // ditto, but 2-byte index
case object EXT4 extends            OptionCode(0x84.toByte) // ditto, but 4-byte index
case object TUPLE1 extends          OptionCode(0x85.toByte) // build 1-tuple from stack top
case object TUPLE2 extends          OptionCode(0x86.toByte) // build 2-tuple from two topmost stack items
case object TUPLE3 extends          OptionCode(0x87.toByte) // build 3-tuple from three topmost stack items
case object NEWTRUE extends         OptionCode(0x88.toByte) // push True
case object NEWFALSE extends        OptionCode(0x89.toByte) // push False
case object LONG1 extends           OptionCode(0x8a.toByte) // push long from < 256 bytes
case object LONG4 extends           OptionCode(0x8b.toByte) // push really big long

/**
 * This object provides methods for converting Python RDD's to Scala RDD's and vice-versa.
 * All conversions are done using (de)serialization and mapPartitions.
 * The currently supported data types/conversion schemes are:
 * 
 * Python                 Scala
 * 
 * 1-byte int             Byte
 * 2-byte int             Short
 * 4-byte int             Int
 * <9-byte long           Long
 * >=9-byte long          BigInt
 * float                  Double
 * Unicode str            String
 * True                   true
 * False                  false
 * None                   None
 * list                   Vector[Any]
 * dict                   HashMap[Any,Any]
 * tuple                  TupleN when 0<len(pyTuple)=N<23, Array[Any] otherwise. All Tuple type parameters are Any.
 * 
 * These conversions work in both directions.
 * Here are some additional conversions from Scala to Python only:
 * 
 * Scala                  Python
 * 
 * Float                  float
 * null                   None
 * Char                   Unicode str
 * <9-byte BigInt         long
 * Array[_]               List
 *
 * Additionally, byte str's in Python are converted to Strings in Scala but not vice-versa.
 * If more data types/conversion schemes are desired, please contact Bobbey Reese at breese@yahoo-inc.com.
 */
object RDDConverter{
  /* 
   * Converts the given Python RDD into a Scala RDD.
   * By "pyRDD", I mean the object returned by, assuming you have called your pyspark RDD "rdd", rdd._jrdd.rdd(), which is
   * fundamentally a batched Scala RDD over Array[Byte].
   */
  def pythonToScala(pyRDD : RDD[Array[Byte]]) : RDD[Any] = {
    def partitionMapFunction(iterator : Iterator[Array[Byte]]) : Iterator[Any] = {
      val deserializer = new PythonToScalaDeserializer()
      iterator.flatMap(byteArray => deserializer.load(byteArray).asInstanceOf[Vector[Any]].iterator)
    }
    pyRDD.mapPartitions(partitionMapFunction)
  }
  
  class LazyIterator(var notSerialized : Iterator[_]) extends Iterator[Array[Byte]]{
    val serializer = new ScalaToPythonSerializer()
    def hasNext = notSerialized.hasNext
    def next() : Array[Byte] = serializer.save(notSerialized.next())
  }
  
  /*
   * Pickle-serializes a Scala RDD so that Python may construct a Python RDD from it. 
   */
  def scalaToPython(scalaRDD : RDD[_]) : RDD[Array[Byte]] = {
    //Not batched at the moment, but I think calling "toJavaRDD" does this anyway, which is done on the
    //Python side.
    def partitionMapFunction(iterator : Iterator[Any]) : Iterator[Array[Byte]] = {
      new LazyIterator(iterator)
    }
    scalaRDD.mapPartitions(partitionMapFunction)
  }
}

//It's just a Byte wrapper.
class OptionCode(val code : Byte){
  override def equals(obj : Any) = obj match {
    //Because I have no way of making an equivalence relation with Bytes symmetric, I have decided to only allow equality between Opcodes.
    case that : OptionCode => code==that.code
    case _ => false
  }
}

object OptionCode{
  implicit def toByte(code : OptionCode) = code.code
}


/**
 * Given a byte array representing a Pickle-serialization of some Python object, a PythonToScalaDeserializer
 * will deserialize it into an equivalent or similar Python object.
 * For the most part, this is a Scala version of this Python to Java Unpickler: 
 * https://github.com/TargetHolding/pyspark-elastic/blob/master/src/main/java/net/razorvine/pickle/custom/Unpickler.java
 * In the future, unit tests will be added to ensure the validity of these conversions.
 */
class PythonToScalaDeserializer{
  case class UnimplementedCodeException (code : String) extends Exception("The option code \""+code+"\" is not implemented."){}
  //marker will only have equality with itself, which is its only purpose.
  private val marker = new Object()
  
  //A stack containing all the currently deserialized Objects.
  private var stack : List[Any] = null
  
  //A HashMap from Int to Any which saves deserialized Objects for later convenience.
  private var memo : HashMap[Int,Any] = null
  
  //A ByteBuffer holding the Bytes from the given Array[Byte].
  private var input : ByteBuffer = null
  
  //Interprets all bytes until the next newline character as characters and returns a String
  //formed from them.
  def readLine() : String = {
    val stringBuffer = new StringBuffer()
    var currentChar = (input.get() & 0xFF).toChar
    while (currentChar != '\n') {
      stringBuffer.append(currentChar)
      //Bit mask necessary to prevent incorrect sign extension when calling toChar,
      //since Scala Bytes are signed. Note that 0xFF is an Int, not a Byte.
      currentChar = (input.get() & 0xFF).toChar
    }
    stringBuffer.toString
  }
  
  //Returns and removes the stack top, aka first list element.
  def pop() : Any = {
    val obj = stack(0)
    stack = stack.tail
    obj
  }
  
  //Pops objects from the stack until the marker is popped.
  def popMark() {
    while (pop() != marker){}
  }
  
  //Forms a Vector from the topmost stack slice.
  //The top element on the stack is the LAST element in the Vector.
  def vectorFromSlice() : Vector[Any] = {
    var vector = Vector[Any]()
    while (stack(0) != marker){
      vector +:= pop()
    }
    //Get rid of the marker.
    stack = stack.tail
    vector
  }
  
  //Appends the object at the stack top to the Vector below it.
  def appendTopToVector() {
    val obj = pop()
    //Roundabout, but necessary if I am going to use an immutable Vector for the Python List equivalent.
    val newVec = pop().asInstanceOf[Vector[Any]] :+ obj
    stack +:= newVec
  }
  
  //Appends the topmost Stack slice to the Vector below it.
  def appendSliceToVector() {
    val vector = vectorFromSlice()
    val newVec = pop().asInstanceOf[Vector[Any]] ++: vector
    stack +:= newVec
  }  
  
  //Pops the two topmost Objects as a Key-Value pair.
  //The stack top is the value.
  def popKeyValuePair() : (Any,Any) = {
    val value = pop()
    val key = pop()
    (key,value)
  }
  
  //Interprets the topmost stack slice as key-value pairs
  //and creates a HashMap from them.
  //The order is value1,key1,value2,key2,...
  def hashMapFromSlice() : HashMap[Any,Any] = {
    val hashMap = HashMap[Any,Any]()
    while (stack(0) != marker){
      hashMap += popKeyValuePair()
    }
    //Get rid of the marker.
    stack = stack.tail
    hashMap
  }
  
  //Interprets the two topmost objects as a key-value pair
  //and puts them in the HashMap below them.
  //The stack top is the value.
  def putTopInHashMap() {
    val keyValuePair = popKeyValuePair()
    stack(0).asInstanceOf[HashMap[Any,Any]] += keyValuePair
  }
  
  //Interprets the topmost stack slice as key-value pairs
  //and puts them in the HashMap below them.
  //The order is value1,key1,value2,key2,...
  def putSliceInHashMap() {
    val sliceMap = hashMapFromSlice()
    stack(0).asInstanceOf[HashMap[Any,Any]] ++= sliceMap
  }
  
  //Interprets the next line of bytes as characters representing a Long and pushes that Long
  //onto the stack.
  //If the Long is larger than 64 bytes, the line is interpreted as a BigInt instead.
  def pushLongFromString() {
    var stringRep = readLine()
    if( stringRep.charAt(stringRep.length()-1) == 'L' ){
      stringRep=stringRep.substring(0,stringRep.length()-1)
    }
    try{
      stack +:= stringRep.toLong
    }
    catch{
      //Assuming the input is not malformed, a NumberFormatException implies the Python Long is
      //more than 64-bit, and thus a BigInt is needed to represent it.
      case ex : NumberFormatException => stack +:= BigInt(stringRep)
    }
      
  }
  
  //Interprets the next numBytes bytes as a little-endian representation of a BigInt and pushes
  //that BigInt onto the stack.
  def pushBinaryBigInt(numBytes : Int) {
    val byteArray = new Array[Byte](numBytes)
    //BigInt expects Big-Endian, bytes are in Little-Endian. Hence, we iterate in reverse.
    for(i <- numBytes-1 to 0 by -1){
      byteArray(i)=input.get()
    }
    stack +:= BigInt(byteArray) 
  }
  
  //Interprets the next numByte bytes as a little-endian representation of a Long and pushes
  //that Long onto the stack.
  //Assumes numBytes <= 8
  def pushPrimitiveBinaryLong(numBytes : Int) {
    val byteArray = new Array[Byte](numBytes)
    input.get(byteArray,0,numBytes)
    var longToPush = 0L
    for (i <- 0 until numBytes - 1){
      
      /* 
       * Interpreting a binary little-endian value manually.
       * Bit-anding with 0xff is necessary because Scala bytes are signed,
       * so extending a "negative" byte to long causes the extended
       * representation to have an unwanted mask, i.e., 0x<first nibble><second nibble> becomes 0xffffffffffffff<first nibble><second nibble>
       * instead of 0x00000000000000<first nibble><second nibble> whenever first nibble is at least 8.
       */
      
      longToPush |= (byteArray(i).toLong & 0xff) << i * 8
    }
    //If the last byte is negative, we don't want to bit-and with 0xff because then we would lose the negative sign whenever numBytes is less than 8.
    longToPush |= byteArray(numBytes - 1).toLong << (numBytes - 1) * 8
    stack +:= longToPush
  }
  
  //Interprets the next numByte bytes as a little-endian representation of some integer.
  //If numBytes is less than 9, the result is pushed as a Long.
  //Otherwise, the result is pushed as a BigInt.
  def pushBinaryLong(numBytes : Int) {
    if(numBytes==0){
      //Apparently cPickle will actually tell you to parse 0 bytes if you push a 0.
      stack +:= 0L
    }
    else if (numBytes<9){
      pushPrimitiveBinaryLong(numBytes)
    }
    else{
      pushBinaryBigInt(numBytes)
    }
  }
  
  //Interprets the next 8 bytes as a Double and pushes it onto the stack.
  def pushDouble() {
    //Either Python or ByteBuffer is inconsistent about the byte ordering for floats,
    //so I need to change it in this method.
    input.order(ByteOrder.BIG_ENDIAN)
    stack +:= input.getDouble()
    input.order(ByteOrder.LITTLE_ENDIAN)
  }
  
  //Interprets the next numBytes bytes as a UTF-8 encoding and pushes it onto the
  //stack as a String.
  def pushBinaryUnicode(numBytes : Int){
    val bytes = new Array[Byte](numBytes)
    input.get(bytes,0,numBytes)
    stack +:= new String(bytes,"UTF-8")
  }
  
  //Creates a Tuple1 from the topmost stack element and pushes it onto the stack.
  def pushTuple1() {
    val tup = Tuple1(pop())
    stack +:= tup
  }
  
  //Creates a Tuple2 from the two topmost stack elements and pushes it onto the stack.
  //The topmost element is the last item in the Tuple.
  def pushTuple2() {
    val second = pop()
    val tup = (pop(),second)
    stack +:= tup
  }
  
  //Creates a Tuple3 form the three topmost stack elements and pushes it onto the stack.
  //The topmost element is the last item in the Tuple.
  def pushTuple3() {
    val third = pop()
    val second = pop()
    val tup = (pop(),second,third)
    stack +:= tup
  }
  
  //Creates a Tuple of the appropriate size from the topmost stack slice.
  //If the topmost stack slice has size outside of [1,22], then an array of Any
  //is pushed instead.
  def tupleFromVector() {
    //Encumbering, but since Tuples in Scala may only go as high as 22, we may as well address each case.
    val vector = vectorFromSlice()
    stack +:= (vector.size match {
      case 1 => Tuple1(vector(0))
      case 2 => Tuple2(vector(0),vector(1))
      case 3 => Tuple3(vector(0),vector(1),vector(2))
      case 4 => Tuple4(vector(0),vector(1),vector(2),vector(3))
      case 5 => Tuple5(vector(0),vector(1),vector(2),vector(3),vector(4))
      case 6 => Tuple6(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5))
      case 7 => Tuple7(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6))
      case 8 => Tuple8(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7))
      case 9 => Tuple9(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8))
      case 10 => Tuple10(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9))
      case 11 => Tuple11(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10))
      case 12 => Tuple12(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11))
      case 13 => Tuple13(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12))
      case 14 => Tuple14(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13))
      case 15 => Tuple15(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14))
      case 16 => Tuple16(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15))
      case 17 => Tuple17(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15),vector(16))
      case 18 => Tuple18(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15),vector(16),vector(17))
      case 19 => Tuple19(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15),vector(16),vector(17),vector(18))
      case 20 => Tuple20(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15),vector(16),vector(17),vector(18),vector(19))
      case 21 => Tuple21(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15),vector(16),vector(17),vector(18),vector(19),vector(20))
      case 22 => Tuple22(vector(0),vector(1),vector(2),vector(3),vector(4),vector(5),vector(6),vector(7),vector(8),vector(9),vector(10),vector(11),vector(12),vector(13),vector(14),vector(15),vector(16),vector(17),vector(18),vector(19),vector(20),vector(21))
      case _ => vector.toArray //If the Vector's size is greater than 22 or 0, return an Any array instead. Obviously, this situation is very uncommon.
    })
  }
  
  //Interprets byteArray as a pickle serialization of a Python object and returns
  //a functionally similar Scala object of that object.
  def load(byteArray : Array[Byte]) : Any = {
    println(byteArray.toString)
    input = ByteBuffer.wrap(byteArray)
    stack = List[Any]()
    memo = HashMap[Int,Any]()
    
    input.order(ByteOrder.LITTLE_ENDIAN) //Python serializes in little-endian, while Java is by default big-endian.
    
    /* while(true) is efficient and elegant here, screw the system!
     * The halting condition is case STOP (second case below).
     * Many codes are not implemented so that I could get this on the floor without
     * having to address uncommon use cases which are absent from the vast majority of our
     * applications. These codes will cause an UnimplementedCodeException to be thrown.
     * In particular, none of Protocol 3 and 4 are supported (see http://grepcode.com/file/repo1.maven.org/maven2/org.spark-project/pyrolite/2.0.1/net/razorvine/pickle/Opcodes.java for a description of these Protocols).
     */
    while(true){
      new OptionCode(input.get()) match {
        //IMPORTANT: Compound assignments are often avoided because if stack on the RHS is parsed BEFORE a function that alters it,
        //these changes will not be reflected in the final result because Scala has already copied an address to an immutable object!
        case MARK             => stack +:= marker
        
        case STOP             => return stack(0) //Terminating condition.
        
        case POP              => stack = stack.tail
        
        case POP_MARK         => popMark()
        
        case DUP              => stack +:= stack(0)
        
        case FLOAT            => stack +:= readLine().toDouble
        
        case INT              => pushLongFromString() //It would SEEM that this option is used when the system architecture has ints as 
                                                      //64-bit (i.e., how Python will acknowledge them), so the argument should be parsed as a Long.
        
        case BININT           => stack +:= input.getInt()//BININT refers to 4 bytes.
        
        case BININT1          => stack +:= input.get()//Here, we WANT a 1 byte signed int, so we don't bit-and with 0xFF
        
        case LONG             => pushLongFromString() //If under 9 bytes, pushes long. Otherwise, pushes BigInt.
        
        case BININT2          => stack +:= input.getShort()
        
        case NONE             => stack +:= None
        
        case PERSID           => throw UnimplementedCodeException("PERSID")
        
        case BINPERSID        => throw UnimplementedCodeException("BINPERSID")
        
        case REDUCE           => throw UnimplementedCodeException("REDUCE")
        
        case STRING           => stack +:= readLine()
        
        case BINSTRING        => pushBinaryUnicode(input.getInt())
        
        case SHORT_BINSTRING  => pushBinaryUnicode(input.get() & 0xFF)
        
        case UNICODE          => throw UnimplementedCodeException("UNICODE")
        
        case BINUNICODE       => pushBinaryUnicode(input.getInt())
        
        case APPEND           => appendTopToVector()
        
        case BUILD            => throw UnimplementedCodeException("BUILD")
        
        case GLOBAL           => throw UnimplementedCodeException("GLOBAL")
        
        case DICT             => stack = hashMapFromSlice() +: stack
          
        case EMPTY_DICT       => stack = HashMap[Any,Any]() +: stack
          
        case APPENDS          => appendSliceToVector()
          
        case GET              => stack = memo(readLine().toInt) +: stack
          
        case BINGET           => stack = memo(input.get() & 0xFF) +: stack
          
        case INST             => throw UnimplementedCodeException("INST")
          
        case LONG_BINGET      => stack = memo(input.getInt()) +: stack
          
        case LIST             => stack = vectorFromSlice() +: stack
          
        case EMPTY_LIST       => stack = Vector[Any]() +: stack
          
        case OBJ              => throw UnimplementedCodeException("Object")
          
        case PUT              => memo += Tuple2(readLine().toInt,stack(0))
          
        case BINPUT           => memo += Tuple2(input.get() & 0xFF,stack(0))
          
        case LONG_BINPUT      => memo += Tuple2(input.getInt(),stack(0))
          
        case SETITEM          => putTopInHashMap()
          
        case TUPLE            => tupleFromVector()
          
        case EMPTY_TUPLE      => stack = new Array[Any](0) +: stack
          
        case SETITEMS         => putSliceInHashMap()
          
        case BINFLOAT         => pushDouble()
          
        case PROTO            => input.get() //We don't really care what the protocol is, so we discard it.
          
        case NEWOBJ           => throw UnimplementedCodeException("NEWOBJ")
          
        case EXT1             => throw UnimplementedCodeException("EXT1")
          
        case EXT2             => throw UnimplementedCodeException("EXT2")
          
        case EXT4             => throw UnimplementedCodeException("EXT4")
          
        case TUPLE1           => pushTuple1()
          
        case TUPLE2           => pushTuple2()
          
        case TUPLE3           => pushTuple3()
          
        case NEWTRUE          => stack +:= true
          
        case NEWFALSE         => stack +:= false
          
        case LONG1            => pushBinaryLong(input.get() & 0xFF) //Here, the "1" does not mean we are pushing a 1-byte long, but rather that 1 byte is used to represent the amount of bytes in the long, i.e., <256 total bytes.
                                                              //When this number is less than 9, a Long is pushed to the stack. Otherwise, a BigInt is pushed to the stack.
          
        case LONG4            => pushBinaryLong(input.getInt()) //I do not know whether this opCode guarantees that the long is at least 256 bytes, so the same check for LONG1 is made to see if a Long or BigInt
                                                                 //should be pushed.
          
        case other            => throw UnimplementedCodeException(other.toChar.toString)
      }
    }
    
  }
}

/**
 * Given a supported Scala object or Iterator[Any], ScalaToPythonSerializer will Pickle the Scala object so Python
 * may deserialize it.
 * In the future, unit tests will be used to test the validity of the serializations.
 */
class ScalaToPythonSerializer{
  //For unimplemented objects, i.e., anything that PythonToScalaDeserializer wouldn't map to.
  case class UnimplementedObjectException(any : Any) extends Exception("Scala to Python serialization is unimplemented for "+any.getClass().getName()+".")
  
  //Holds the currently serialized bytes.
  //An ArrayBuffer rather than a ByteBuffer was used for convenience: A ByteBuffer
  //may have to be resized manually.
  var bytes : ArrayBuffer[Byte] = null
  
  //Adds the Pickle-serialization of a single Byte to the aggregate serialization.
  def dumpByte(byte : Byte){
    bytes += BININT1
    bytes += byte
  }
  
  //Adds the Pickle-serialization of a binary Char to the aggregate serialization, which is interpreted as a two-byte unicode character.
  def dumpChar(char : Char){
    bytes += BINUNICODE
    bytes += char.toByte
    bytes += (char.toShort >> 8).toByte
  }
  
  //Adds the Pickle-serialization of a binary Short to the aggregate serialization.
  def dumpShort(short : Short){
    bytes += BININT2
    bytes += (0xff & short).toByte
    bytes += (0xff & short>>8).toByte
  }
  
  //Puts the bytes of an Int  to the aggregate serialization in little-endian order.
  def putInt(int : Int){
    bytes += (0xff & int).toByte
    bytes += (0xff & int>>8).toByte
    bytes += (0xff & int>>16).toByte
    bytes += (0xff & int>>24).toByte
  }
  
  //Adds the Pickle-serialization of a binary Int to the aggregate serialization.
  def dumpInt(int : Int){
    bytes += BININT
    putInt(int)
  }
  
  //Adds the Pickle-serialization of a binary Long to the aggregate serialization.
  def dumpLong(long : Long){
    bytes += LONG1
    bytes += 8.toByte
    bytes += (0xff & long).toByte
    bytes += (0xff & long>>8).toByte
    bytes += (0xff & long>>16).toByte
    bytes += (0xff & long>>24).toByte
    bytes += (0xff & long>>32).toByte
    bytes += (0xff & long>>40).toByte
    bytes += (0xff & long>>48).toByte
    bytes += (0xff & long>>56).toByte
  }
  
  //Adds the Pickle-serialization of a binary BigInt to the aggregate serialization.
  def dumpBigInt(bigInt : BigInt) {
    val byteArray = bigInt.toByteArray
    if(byteArray.length < 256){
      bytes += LONG1
      bytes += byteArray.length.toByte
    }
    else{
      bytes += LONG4
      putInt(byteArray.length)
    }
    for (i <- byteArray.length-1 to 0 by -1){
      bytes += byteArray(i)
    }
  }
  
  //Adds the Pickle-serialization of a String as a unicode utf8 encoding to the aggregate serialization.
  def dumpString(string : String) {
    val utf8 = string.getBytes
    bytes += BINUNICODE
    putInt(utf8.length)
    bytes ++= utf8
  }
  
  //Adds the Pickle-serialization of a Double as an 8-byte floating-point encoding to the aggregate serialization.
  def dumpDouble(double : Double) {
    //Python only has one floating point type. Its size is 8 bytes, or at least, that's how serialization encodes them.
    bytes += BINFLOAT
    val buffer=ByteBuffer.allocate(8)
    buffer.order(ByteOrder.BIG_ENDIAN)
    buffer.putDouble(double)
    buffer.position(0)
    for(i <- 0 to 7){
      bytes += buffer.get()
    }
  }
  
  //Adds the Pickle-serialization of a Python list representative of the given Vector to the aggregate serialization.
  def dumpVector(vector : Vector[_]) {
    bytes += MARK
    for (obj <- vector){
      serialize(obj)
    }
    bytes += LIST
  }
  
  //Adds the Pickle-serialization of a Python dict representative of the given HashMap to the aggregate serialization.
  def dumpHashMap(hashMap : HashMap[_,_]) {
    bytes += MARK
    for (i <- hashMap){
      serialize(i._1)
      serialize(i._2)
    }
    bytes += DICT
  }
  
  //The following 22 methods add the Pickle-serialization of a Python tuple representative of the given TupleN's to the aggregate serialization.
  
  def dumpTuple(tup : Tuple1[_]){
    serialize(tup._1)
    bytes += TUPLE1
  }
  
  def dumpTuple(tup : Tuple2[_,_]){
    serialize(tup._1)
    serialize(tup._2)
    bytes += TUPLE2
  }
  
  def dumpTuple(tup : Tuple3[_,_,_]){
    serialize(tup._1)
    serialize(tup._2)
    serialize(tup._3)
    bytes += TUPLE3
  }
  
  def dumpTuple(tup : Tuple4[_,_,_,_]){
    bytes += MARK
    serialize(tup._1)
    serialize(tup._2)
    serialize(tup._3)
    serialize(tup._4)
    bytes += TUPLE
  }
  
  //For compactness, the rest of the Tuple functions will be written on one line.
  //They are merely extended versions of dumpTuple for Tuple4.
  def dumpTuple(tup : Tuple5[_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);bytes += TUPLE}
  def dumpTuple(tup : Tuple6[_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);bytes += TUPLE}
  def dumpTuple(tup : Tuple7[_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);bytes += TUPLE}
  def dumpTuple(tup : Tuple8[_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);bytes += TUPLE}
  def dumpTuple(tup : Tuple9[_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);bytes += TUPLE}
  def dumpTuple(tup : Tuple10[_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);bytes += TUPLE}
  def dumpTuple(tup : Tuple11[_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);bytes += TUPLE}
  def dumpTuple(tup : Tuple12[_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);bytes += TUPLE}
  def dumpTuple(tup : Tuple13[_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);bytes += TUPLE}
  def dumpTuple(tup : Tuple14[_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);bytes += TUPLE}
  def dumpTuple(tup : Tuple15[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);bytes += TUPLE}
  def dumpTuple(tup : Tuple16[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);bytes += TUPLE}
  def dumpTuple(tup : Tuple17[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);serialize(tup._17);bytes += TUPLE}
  def dumpTuple(tup : Tuple18[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);serialize(tup._17);serialize(tup._18);bytes += TUPLE}
  def dumpTuple(tup : Tuple19[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);serialize(tup._17);serialize(tup._18);serialize(tup._19);bytes += TUPLE}
  def dumpTuple(tup : Tuple20[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);serialize(tup._17);serialize(tup._18);serialize(tup._19);serialize(tup._20);bytes += TUPLE}
  def dumpTuple(tup : Tuple21[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);serialize(tup._17);serialize(tup._18);serialize(tup._19);serialize(tup._20);serialize(tup._21);bytes += TUPLE}
  def dumpTuple(tup : Tuple22[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]){bytes += MARK;serialize(tup._1);serialize(tup._2);serialize(tup._3);serialize(tup._4);serialize(tup._5);serialize(tup._6);serialize(tup._7);serialize(tup._8);serialize(tup._9);serialize(tup._10);serialize(tup._11);serialize(tup._12);serialize(tup._13);serialize(tup._14);serialize(tup._15);serialize(tup._16);serialize(tup._17);serialize(tup._18);serialize(tup._19);serialize(tup._20);serialize(tup._21);serialize(tup._22);bytes += TUPLE}

  //Adds the Pickle-serialization of a Python tuple representative of the given array to the aggregate serialization.
  def dumpArray(array : Array[_]){
    bytes += MARK
    for(obj <- array){
      serialize(obj)
    }
    bytes += LIST
  }
  
  //Pickle-serializes a Scala object into an equivalent or similar Python object.
  def save(any : Any) : Array[Byte] = {
    bytes = ArrayBuffer[Byte](MARK)
    serialize(any)
    bytes += LIST
    bytes += STOP
    bytes.toArray
  }
  
  //Pickle-serializes every Scala object in an Iterator of Scala objects into equivalent or similar Python objects.
  def save(iterator : Iterator[Any]) : Array[Byte] = {
    bytes = ArrayBuffer[Byte](MARK)
    for(obj <- iterator){
      serialize(obj)
    }
    bytes += LIST
    bytes += STOP
    bytes.toArray
  }
  
  //Adds the Pickle-serialization of a Python object representative or equivalent to the given Scala object to the aggregate serialization.
  //Throws an UnimplementedObjectException if serialization of the Scala object is not yet supported.
  def serialize(any : Any) {
    
    any match{
    
      case that : Byte           => dumpByte(that)
      
      case that : Char          => dumpChar(that)
      
      case that : Short         => dumpShort(that)
      
      case that : Int           => dumpInt(that)
      
      case that : Long          => dumpLong(that)
      
      case that : Float          => dumpDouble(that.toDouble)
      
      case that : Double        => dumpDouble(that)
      
      case that : BigInt        => dumpBigInt(that)
      
      case true                  => bytes += NEWTRUE
      
      case false                => bytes += NEWFALSE
      
      case None                  => bytes += NONE
      
      case null                  => bytes += NONE
      
      case that : String        => dumpString(that)    
      
      case that : Vector[_]      => dumpVector(that)
      
      case that : HashMap[_,_]  => dumpHashMap(that)
      
      case that : Array[_]      => dumpArray(that)
      
      case that : Tuple1[_]      => dumpTuple(that)
      
      case that : Tuple2[_,_]      => dumpTuple(that)
      
      case that : Tuple3[_,_,_]      => dumpTuple(that)
      
      case that : Tuple4[_,_,_,_]      => dumpTuple(that)
      
      case that : Tuple5[_,_,_,_,_]      => dumpTuple(that)
      
      case that : Tuple6[_,_,_,_,_,_]      => dumpTuple(that)
      
      case that : Tuple7[_,_,_,_,_,_,_]      => dumpTuple(that)
      
      case that : Tuple8[_,_,_,_,_,_,_,_]      => dumpTuple(that)
      
      case that : Tuple9[_,_,_,_,_,_,_,_,_]      => dumpTuple(that)
      
      case that : Tuple10[_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple11[_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple12[_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple13[_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple14[_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple15[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple16[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple17[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple18[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple19[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple20[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple21[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that : Tuple22[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_]    => dumpTuple(that)
      
      case that                 => throw UnimplementedObjectException(that)    
    }  
  }
}

object General {
  def getTypeParameters(typ : Type) : List[Type]={

        typ match{ case TypeRef(_,_,args) => args
                   case a:ExistentialType => getTypeParameters(a.underlying)}
  }
}


















