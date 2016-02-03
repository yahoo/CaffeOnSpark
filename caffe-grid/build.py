#!/usr/bin/env python

import re, os
from os import walk, path
import sys
import shutil, uuid, StringIO

import glob
import zipfile
from __main__ import StringIO

def zipdir(src, dst, zip):
    for dir, dirs, files in walk(src):
        for file in files:
            zip.write(path.join(dir, file), path.join(dst, file))

if __name__ == '__main__':
    target = glob.glob('target/*.jar')[0]
    jar = 'caffe.jar'
    shutil.copyfile(target, jar)

    zip = zipfile.ZipFile(jar, 'a')
    zipdir('target/lib', 'lib', zip)
    id = uuid.uuid4().hex
    print id
    zip.writestr('build-id', id)
    zip.close()

    if not os.path.exists('jars'):
        os.makedirs('jars')
    for dep in glob.glob("target/dependency/*.jar"):
        # Remove platform specific jars
        if '-macosx-x86_64' not in dep and '-linux-x86_64' not in dep:
            shutil.copyfile(dep, 'jars/' + path.basename(dep))




