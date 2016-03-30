'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''
import subprocess
def toCamelCase(prefix,string):
    return prefix+string[0].upper()+string[1:]

def isPrefix(prefix,string):
    return len(prefix) <= len(string) and string[:len(prefix)]==prefix

def isSuffix(suffix,string):
    return len(suffix) <= len(string) and string[-len(suffix):]==suffix

def unCamelCase(prefixLength,string):
    if prefixLength==len(string):
        return ""
    return string[prefixLength].lower() + string[prefixLength+1:]

def getTrailingNumber(string):
    i=len(string)-1
    while string[i].isdigit():
        i-=1
    if i==len(string)-1:
        return 0
    return int(string[i+1:])

def deleteFile(path,isHadoop=False):
    commandStr = "rm -r "+path
    if isHadoop:
        commandStr="hadoop fs -" + commandStr
    subprocess.Popen(commandStr,shell=True).communicate()
