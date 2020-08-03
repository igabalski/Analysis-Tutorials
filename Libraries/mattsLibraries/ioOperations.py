import sys
import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def findPath(name):
    for path in sys.path:
        thePath = find(name,path)
        if thePath is not None:
            return thePath

def removeIgnoredElements(alist, ignorelist):
    for igEl in ignorelist:
        try:
            alist.pop( alist.index(igEl) )
        except ValueError as e:
            if igEl in str(e):
                pass
            else:
                raise ValueError(e)
    return alist

def getSubDirectories(topLevel, ignore=['.git','.ipynb_checkpoints']):
    subDirectories = [o for o in os.listdir(topLevel)
                    if os.path.isdir(os.path.join(topLevel,o))]
    subDirectories = removeIgnoredElements(subDirectories, ignore)
    subDirectories = [topLevel+'/'+sd for sd in subDirectories]


    subsubDirectories = []
    if len(subDirectories) > 0:
        for subDir in subDirectories:
            subSubDirs = getSubDirectories(subDir)
            subsubDirectories.extend(subSubDirs)
    subDirectories.extend(subsubDirectories)
    return subDirectories

def addSubDirectoriesToPath(topLevel, ignore=['.git','.ipynb_checkpoints']):
    subDirectories = getSubDirectories(topLevel, ignore)
    subDirectories.extend(topLevel)
    for sd in subDirectories:
        sys.path.insert(0, sd)
