import numpy as np
import os
import multiprocessing
import matplotlib.pyplot as plt
import shutil

"""
To do: function to link boundary points

"""


class ParamSpace1D:
    def __init__(self, func, p_vals, direc, parallel=False, cores=None):
        """
        Runs func with all combinations of p1_vals and p2_vals, saves results to csv file

        :param func: function, takes two parameter values, returns a float
        :param p_vals: array of parameter values
        :param direc: directory to save results
        :param parallel: if True, will run in parallel using number of cores specified
        :param cores: number of cores to use, if parallel=True

        To do:
        - check that this works
        - ability to stop and resume
        - make more like 2D class, i.e. ability to explore boundaries

        """
        self.func = func
        self.p_vals = p_vals
        self.cores = cores
        self.direc = direc
        self.parallel = parallel

    def single_eval(self, p_val):

        # Run function
        res = self.func(p_val)

        # Save results
        with open(self.direc + '/Res.csv', 'a') as f:
            f.write("{:.12f}".format(p_val) + ',' + str(res) + '\n')

    def run(self):

        # Make results directory, if it doesn't already exist
        if not os.path.isdir(self.direc):
            os.mkdir(self.direc)

        # Run
        if self.parallel:
            pool = multiprocessing.Pool(self.cores)
            pool.map(self.single_eval, self.p_vals)
        else:
            for p in iter(self.p_vals):
                self.single_eval(p)


class ParamSpace2D:
    def __init__(self, func, p1_range, p2_range, resolution0, direc, resolution_step=2, n_iterations=1,
                 explore_boundaries=True, parallel=False, cores=None, crange=None, cmap=None, save_fig=False, args=[],
                 replace=False):
        """

        Class to perform parameter space sweeps

        Run by calling run function
        - performs analysis, saves results with each iteration to .csv files, and saves final figure
        - progress is saved, if interrupted can be resumed without loss by calling the run function again

        Computation parameters
        :param func: function - takes 2 parameters, returns an integer (must not be zero)
        :param p1_range: range for parameter 1 (lower, upper)
        :param p2_range: range for parameter 2 (lower, upper)
        :param resolution0: n x n points on initial grid
        :param resolution_step: how much resolution increases with each iteration
        :param n_iterations: number of iterations
        :param explore_boundaries: if True, will focus parameter search to regions where func result varies
        :param parallel: if True, will run in parallel using number of cores specified
        :param cores: number of cores on machine to use in parallel
        :param args: additional arguments for func

        # Saving parameters
        :param direc: directory to save results. Directory must already exist

        # Figure parameters
        :param save_fig: if True, will save figure
        :param crange: (lowest value, highest value)
        :param cmap: if None, pyplot will use 'viridis'

        To do:
        - adjust so it can be run from multiple machines simultaneously
        - test

        """

        # Computation
        self.func = func
        self.p1_range = p1_range
        self.p2_range = p2_range
        self.resolution0 = resolution0
        self.resolution_step = resolution_step
        self.n_iterations = n_iterations
        self.explore_boundaries = explore_boundaries
        self.parallel = parallel
        self.args = args

        if cores is None:
            self.cores = multiprocessing.cpu_count()
        else:
            self.cores = cores

        # Saving
        self.direc = direc
        self.save_fig = save_fig
        self.replace = replace

        # Figure
        if crange is None:
            self.crange = [None, None]
        else:
            self.crange = crange
        self.cmap = cmap

        # Results
        self.iteration = None
        self.res = None
        self.n_sims = None

    def single_eval(self, p1val_p2val):
        """
        Single function call for given p1 and p2 values, save result

        """

        # Run function
        state = self.func(*[float(i) for i in p1val_p2val.split(',')] + self.args)

        # Save state
        with open(self.direc + '/' + str(self.iteration) + '.csv', 'a') as f:
            f.write(p1val_p2val + ',' + str(state) + '\n')

    def batch_eval(self, pcombs):
        """
        Evaluate parameter sets in bulk
        pcombs is list of strings 'p1val,p2val'

        """
        if self.parallel:
            pool = multiprocessing.Pool(self.cores)
            pool.map(self.single_eval, pcombs)
        else:
            for k in iter(pcombs):
                self.single_eval(k)

    def import_res(self):
        """
        Import all results from current iteration, load into self.res

        """

        with open(self.direc + '/' + str(self.iteration) + '.csv') as g:
            for line in g:
                p1, p2, val = line[:-1].split(',')
                xind = ((float(p1) - self.p1_range[0]) * (self.n_sims - 1)) / (self.p1_range[1] - self.p1_range[0])
                yind = ((float(p2) - self.p2_range[0]) * (self.n_sims - 1)) / (self.p2_range[1] - self.p2_range[0])
                if '.' in val:
                    self.res[round(xind), round(yind)] = float(val)
                else:
                    self.res[round(xind), round(yind)] = int(val)

    def run(self):
        """
        Run algorithm, save figure

        """

        # Clear results directory if replace is True
        if self.replace:
            if os.path.isdir(self.direc):
                shutil.rmtree(self.direc)

        # Make results directory, if it doesn't already exist
        if not os.path.isdir(self.direc):
            os.mkdir(self.direc)

        for iteration in range(self.n_iterations):
            print(iteration)
            self.iteration = iteration

            # First iteration, initial grid
            if self.iteration == 0:
                self.n_sims = self.resolution0
                run_bool = np.ones([self.n_sims, self.n_sims])

            # Subsequent iteration, explore boundary regions
            else:
                self.n_sims = self.resolution_step * (self.n_sims - 1) + 1

                if self.explore_boundaries:
                    # Find boundary regions
                    a = np.nonzero(np.nan_to_num(np.diff(self.res, axis=0)))
                    b = np.nonzero(np.nan_to_num(np.diff(self.res, axis=1)))
                    c = np.nonzero(np.nan_to_num(self.res[:-1, :-1] - self.res[1:, 1:]))
                    xpoints = np.r_[a[0], b[0], c[0]]
                    ypoints = np.r_[a[1], b[1], c[1]]
                    run_bool = np.zeros([self.n_sims, self.n_sims])
                    for x, y in zip(xpoints, ypoints):
                        run_bool[x * self.resolution_step:x * self.resolution_step + (self.resolution_step + 1),
                        y * self.resolution_step:y * self.resolution_step + (self.resolution_step + 1)] = 1

                else:
                    run_bool = np.ones([self.n_sims, self.n_sims])

            # Parameter combinations
            sims_array_ind = np.nonzero(run_bool)
            p1vals = self.p1_range[0] + sims_array_ind[0] * (self.p1_range[1] - self.p1_range[0]) / (self.n_sims - 1)
            p2vals = self.p2_range[0] + sims_array_ind[1] * (self.p2_range[1] - self.p2_range[0]) / (self.n_sims - 1)
            pcombs = ["{:.12f}".format(p1vals[i]) + ',' + "{:.12f}".format(p2vals[i]) for i in range(len(p1vals))]

            # Remove parameters already tested (if algorithm run before)
            if os.path.isfile(self.direc + '/' + str(self.iteration) + '.csv'):
                with open(self.direc + '/' + str(self.iteration) + '.csv') as f:
                    for line in f:
                        p = line.split(',')[0] + ',' + line.split(',')[1]
                        if p in pcombs:
                            pcombs.remove(p)

            # Carry over combinations from previous iteration
            if self.iteration != 0:
                with open(self.direc + '/' + str(self.iteration - 1) + '.csv') as f:
                    with open(self.direc + '/' + str(self.iteration) + '.csv', 'a') as g:
                        for line in f:
                            p = line.split(',')[0] + ',' + line.split(',')[1]
                            if p in pcombs:
                                pcombs.remove(p)
                                g.write(line)

            # Run
            self.batch_eval(pcombs)

            # Import results
            self.res = np.nan * np.zeros([self.n_sims, self.n_sims])
            self.import_res()

            # Safety check: test any untested points directly adjacent to boundaries
            # (often required if resolution0 or resolution_step are too small)
            if self.explore_boundaries and self.iteration != 0:
                j = 1
                while j != 0:

                    # Compare each nan value to all neighbours
                    # (this is quite slow, quicker way?)
                    # throws warning: "All-NaN axis encountered" - this is fine
                    res_padded = np.nan * np.zeros([self.n_sims + 2, self.n_sims + 2])
                    res_padded[1:-1, 1:-1] = self.res
                    x = np.dstack(
                        (res_padded[:-2, :-2], res_padded[:-2, 1:-1], res_padded[:-2, 2:], res_padded[1:-1, :-2],
                         res_padded[1:-1, 2:], res_padded[2:, :-2], res_padded[2:, 1:-1], res_padded[2:, 2:]))

                    mx = np.nanmax(x, axis=2)
                    mn = np.nanmin(x, axis=2)
                    run_bool = (mx == mx) * (mx != mn) * (self.res != self.res)
                    sims_array_ind = np.nonzero(run_bool)
                    j = len(sims_array_ind[0])

                    # Parameter combinations (if any)
                    if j != 0:
                        p1vals = self.p1_range[0] + sims_array_ind[0] * (self.p1_range[1] - self.p1_range[0]) / (
                                self.n_sims - 1)
                        p2vals = self.p2_range[0] + sims_array_ind[1] * (self.p2_range[1] - self.p2_range[0]) / (
                                self.n_sims - 1)
                        pcombs = ["{:.12f}".format(p1vals[i]) + ',' + "{:.12f}".format(p2vals[i]) for i in
                                  range(len(p1vals))]

                        # Run
                        self.batch_eval(pcombs)

                        # Compile results
                        self.import_res()

        # Interpolate missing values
        if self.explore_boundaries:

            # If no boundary (i.e. uniform parameter space), reload output from first iteration
            if np.sum(~np.isnan(self.res)) == 0:
                with open(self.direc + '/0.csv') as f:
                    self.res[:, :] = int(f.readline().split(',')[2])

            # Else: Interpolate nans by flood fill algorithm
            else:
                o = np.argwhere(self.res != self.res)
                while len(o) != 0:
                    pos = o[0]
                    fillval = findzone(self.res, pos[0], pos[1])
                    floodfill(self.res, pos[0], pos[1], fillval)
                    o = np.argwhere(self.res != self.res)

        # Save figure
        if self.save_fig:
            fig, ax = self.figure()
            fig.savefig(self.direc + '/fig.png', dpi=300)
            fig.close()

        # Specify int/float format
        f = (self.res % 1) == 0

        # Save row by row
        with open(self.direc + '/Res.txt', 'w') as fh:
            for i, row in enumerate(self.res):
                formats = f[i]
                line = ' '.join(
                    "{:.0f}".format(value) if formats[j] else "{:.12f}".format(value) for j, value in enumerate(row))
                fh.write(line + '\n')

    def figure(self):
        """
        Parameter space plot

        """

        # Set up
        fig, ax = plt.subplots()

        # Colours
        ax.imshow(self.res.T, origin='lower', aspect='auto', vmin=self.crange[0], vmax=self.crange[1],
                  cmap=self.cmap, extent=(self.p1_range[0], self.p1_range[1], self.p2_range[0], self.p2_range[1]))

        # Figure adjustments
        ax.set_xlim(self.p1_range[0], self.p1_range[1])
        ax.set_ylim(self.p2_range[0], self.p2_range[1])
        fig.set_size_inches(4, 4)
        fig.tight_layout()
        return fig, ax


def floodfill(array, x, y, newval):
    """
    Queue based flood fill algorithm
    Edits array in place

    :param array:
    :param x: x position
    :param y: y position
    :param newval: new value
    :return:
    """

    array[x, y] = newval
    Q = [(x, y)]
    while len(Q) != 0:
        x, y = Q.pop(0)

        if x > 0:
            if np.isnan(array[x - 1, y]):
                array[x - 1, y] = newval
                Q.append((x - 1, y))
        if x < len(array[:, 0]) - 1:
            if np.isnan(array[x + 1, y]):
                array[x + 1, y] = newval
                Q.append((x + 1, y))
        if y > 0:
            if np.isnan(array[x, y - 1]):
                array[x, y - 1] = newval
                Q.append((x, y - 1))
        if y < len(array[0, :]) - 1:
            if np.isnan(array[x, y + 1]):
                array[x, y + 1] = newval
                Q.append((x, y + 1))


def findzone(array, x, y):
    """
    For 2D phase space class.
    Finds zone corresponding to a given location
    Based on flood fill algorithm

    :param array:
    :param x: x position
    :param y: y position
    :return: zone value
    """

    tested = np.zeros(array.shape)
    tested[x, y] = 1
    Q = [(x, y)]
    val = np.nan
    while val != val:
        x, y = Q.pop(0)

        vals = []
        if x > 0:
            if tested[x - 1, y] == 0:
                tested[x - 1, y] = 1
                vals.append(array[x - 1, y])
                Q.append((x - 1, y))
        if x < len(array[:, 0]) - 1:
            if tested[x + 1, y] == 0:
                tested[x + 1, y] = 1
                vals.append(array[x + 1, y])
                Q.append((x + 1, y))
        if y > 0:
            if tested[x, y - 1] == 0:
                tested[x, y - 1] = 1
                vals.append(array[x, y - 1])
                Q.append((x, y - 1))
        if y < len(array[0, :]) - 1:
            if tested[x, y + 1] == 0:
                tested[x, y + 1] = 1
                vals.append(array[x, y + 1])
                Q.append((x, y + 1))

        if len(vals) != 0:
            val = np.nanmax(vals)
    return val
