import numpy as np
import matplotlib.pyplot as plt


def raster(event_times_list, **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, **kwargs)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax

if __name__ == '__main__':
    # example usage
    # Generate test data
    nbins = 100
    ntrials = 10
    spikes = []
    for i in range(ntrials):
        spikes.append(np.arange(nbins)[np.random.rand(nbins) < .2])

    fig = plt.figure()
    ax = raster(spikes)
    plt.title('Example raster plot')
    plt.xlabel('time')
    plt.ylabel('trial')
    fig.show()
    while True:
        pass
