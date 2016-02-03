#!/usr/bin/env python
"""
Given an strace output of a program, this file
copies all of the Shared library and Python libraries
that the program depends on into the $CWD/lib directory.

E.g. strace -o straceoutput.txt -e trace=open -f java ...
"""

import re
import os
import sys
import shutil


def extract_opened_file(line):
    opened_file_re = re.compile(r"^.*open\(\"([^\"]*)")
    match = opened_file_re.match(line)
    if match is not None:
        return match.group(1)
    return None


def is_shared_library(fpath):
    so_library = re.compile(r"^.*\.so.*$")
    match = so_library.match(fpath)
    if match is not None:
        return True
    return False
 
 
def is_etc_ld_so_cache(fpath):
    return "/etc/ld.so.cache" in fpath


def is_jvm_package(fpath):
    return "jdk" in fpath


def find_deps():
    strace_fh = open(sys.argv[1])
 
    for line in strace_fh:
        line = line.strip()
 
        fpath = extract_opened_file(line)
 
        if fpath is None:
            continue
 
        # filter all files not shared library
        if not is_shared_library(fpath):
            continue
 
        # filter out shared library cache
        if is_etc_ld_so_cache(fpath):
            continue
 
        # filter out jvm files
        if is_jvm_package(fpath):
            continue
        
        # filter out non existing files (Cannot rely on ENOENT
        # since some open are marked "<unfinished ...>" then
        # resumed later
        if not os.path.isfile(fpath):
            continue

        yield fpath


def copy_shared_libs():
    try:
        os.makedirs("target/lib")
    except OSError:
        pass
    shared_libraries = list(find_deps())
    for lib in shared_libraries:
        shutil.copy(lib, "target/lib/")
        print "copying %s" % lib


if __name__ == '__main__':
    copy_shared_libs()
