'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

'''
This module simply sets sc,sc._jvm, and sc._gateway
as "builtin" variables sc,jvm, and gateway respectively.
'''
import __builtin__

def registerContext(sc):
    __builtin__.sc=sc
    __builtin__.jvm=sc._jvm
    __builtin__.gateway=sc._gateway

def registerSQLContext(sqlContext):
    __builtin__.sqlContext=sqlContext
