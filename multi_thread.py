import multiprocessing as mp
from functools import partial


def RunInParallel(fns: list, fn_names: list):
    """ Run ANY functions in parallel. Functions to parallelise have to load the args already
        using some method, such as functools.partial. See the usage below.

    - Usage
    ```python
        from functools import partial  # this is to load the args onto a function without execution

        def fn1(a):
            return a, "hello"

        def fn2(a):
            return a, " world"

        fn_names = ["oen", "two"]
        fns = [partial(fn1, a=1), partial(fn2, a=2)]

        res = RunInParallel(fn_names=fn_names, fns=fns)
        print(res)
    ```

    - References
        - https://docs.python.org/3/library/multiprocessing.html
        - https://stackoverflow.com/a/10415215/9246727
        - https://stackoverflow.com/a/7207336/9246727
    """

    # Prep the shared dictionary to write all the result on
    return_dict = mp.Manager().dict()
    proc = list()

    if fn_names is None:
        # fn_names = [i for i in range(len(fns))]
        fn_names = range(len(fns))

    # Assign functions to processes
    for _pid, fn in zip(fn_names, fns):

        def _worker(return_dict):
            """ Base function to execute

                This takes the shared dictionary(in multiprocessing) and write the result of executing a function
                on the shared dictionary.
            """
            result = fn()
            return_dict[_pid] = result

        p = mp.Process(target=_worker, args=(return_dict,))
        p.start()
        proc.append(p)

    # Call all the spawned processes
    for p in proc:
        p.join()

    return return_dict  # At this point, this shared dictionary is being writen by all the spawned process!


def _test():
    import time

    def fn1(a):
        time.sleep(0.5)
        return a, "hello"

    def fn2(a):
        time.sleep(1.5)
        return a, " world"

    fns = [partial(fn1, a=1), partial(fn2, a=2)]
    fn_names = ["oen", "two"]

    res = RunInParallel(fn_names=fn_names, fns=fns)
    print(res)


def _IPS_mapReduce_test():
    """ Test fn to see if we should do map_reduce for IPS

        Seems like the numpy's natural division is much faster than map reduced division...
    """
    import time
    import numpy as np

    def fn(_a, _b):
        print(_a.shape, _b.shape)
        return _a, _b

    mat_a = np.random.rand(1_000_000, 100)
    mat_b = np.random.rand(1_000_000, 100)

    start = time.time()
    res = mat_a / mat_b
    print(time.time() - start)

    # Map Reduce test
    num_split = 3

    fn_names = [str(i) for i in range(num_split)]
    fns = [partial(fn, _a, _b) for _a, _b in zip(np.array_split(mat_a, num_split), np.array_split(mat_b, num_split))]

    start = time.time()
    res = RunInParallel(fn_names=fn_names, fns=fns)
    print(time.time() - start)
    # print(res)


if __name__ == '__main__':
    _test()
    _IPS_mapReduce_test()
