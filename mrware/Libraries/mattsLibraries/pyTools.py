

def ignore_kwargs(kwargs, toggle):
    if (kwargs is not None) and (toggle):
        print 'Ignoring undefined input variable ...'
        print kwargs.keys()



def merge_dictionaries( *args ):
    z = {}
    for arg in args:
        z.update( arg )
    return z
