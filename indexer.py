def get_index_of_min(histogram):
    import numpy as np

    # make sure data is in a standard list, not a numpy array
    if (type(histogram).__module__ == np.__name__):
        histogram = list(histogram)

    # return a list of the indexes of the minimum values. Important if there is >1 minimum
    return [i for i, x in enumerate(histogram) if x == min(histogram)]


def get_index_of_max(histogram):
    import numpy as np

    # make sure data is in a standard list, not a numpy array
    if (type(histogram).__module__ == np.__name__):
        histogram = list(histogram)

    # return a list of the indexes of the max values. Important if there is >1 maximum
    return [i for i, x in enumerate(histogram) if x == max(histogram)]
