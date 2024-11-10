import copy
import copyreg
from collections import OrderedDict
""" From Stack Overflow:
https://stackoverflow.com/a/73226990

"""
def convert_nested_ordered_dict(x):
    """
    Perform a deep copy of the given object, but convert
    all internal OrderedDicts to plain dicts along the way.

    Args:
        x: Any pickleable object

    Returns:
        A copy of the input, in which all OrderedDicts contained
        anywhere in the input (as iterable items or attributes, etc.)
        have been converted to plain dicts.
    """
    # Temporarily install a custom pickling function
    # (used by deepcopy) to convert OrderedDict to dict.
    orig_pickler = copyreg.dispatch_table.get(OrderedDict, None)
    copyreg.pickle(
        OrderedDict,
        lambda d: (dict, ([*d.items()],))
    )
    try:
        return copy.deepcopy(x)
    finally:
        # Restore the original OrderedDict pickling function (if any)
        del copyreg.dispatch_table[OrderedDict]
        if orig_pickler:
            copyreg.dispatch_table[OrderedDict] = orig_pickler
