import argparse
import os
import pathlib
from multiprocessing import Process

from npbench.infrastructure import (
    Benchmark,
    generate_framework,
    LineCount,
    Test,
    utilities as util,
)

def run_benchmark(benchname, framework_name, preset, validate, repeat,
                  timeout, ignore_errors, save_strict, load_strict):
    frmwrk = generate_framework(framework_name,
                                save_strict=save_strict,
                                load_strict=load_strict)
    numpy = generate_framework("numpy")
    bench = Benchmark(benchname)
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    # Test.run signature in your first script also takes datatype; if you need it, add it.
    test.run(preset, validate, repeat, timeout,
             ignore_errors=ignore_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--preset",
                        choices=["S", "M", "L", "paper"],
                        nargs="?",
                        default="S")
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-t",
                        "--timeout",
                        type=float,
                        nargs="?",
                        default=200.0)
    parser.add_argument("--ignore-errors",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-s",
                        "--save-strict-sdfg",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    parser.add_argument("-l",
                        "--load-strict-sdfg",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    args = vars(parser.parse_args())

    # 1) Discover all benchmarks (54 JSONs) from bench_info
    parent_folder = pathlib.Path(__file__).parent.absolute()
    bench_dir = parent_folder.joinpath("bench_info")
    pathlist = pathlib.Path(bench_dir).rglob("*.json")
    benchnames = [os.path.basename(path)[:-5] for path in pathlist]
    benchnames.sort()

    # 2) List all frameworks you care about
    # IMPORTANT: adjust these strings to whatever `generate_framework` expects
    frameworks = [
        "numpy",
        "numba",
        "cupy",
        # "jax",
        "pythran",
        # "dpnp",
        "triton",
        "dace_cpu",
        "dace_gpu",
    ]

    failed = []

    # 3) Run every (benchmark, framework) pair
    for benchname in benchnames:
        for framework_name in frameworks:
            print(f"=== {benchname} / {framework_name} ===")
            p = Process(
                target=run_benchmark,
                args=(
                    benchname,
                    framework_name,
                    args["preset"],
                    args["validate"],
                    args["repeat"],
                    args["timeout"],
                    args["ignore_errors"],
                    args["save_strict_sdfg"],
                    args["load_strict_sdfg"],
                ),
            )
            p.start()
            p.join()
            exit_code = p.exitcode
            if exit_code != 0:
                failed.append((benchname, framework_name))

    if failed:
        print(
            f"Failed: {len(failed)} (benchmark, framework) pairs "
            f"out of {len(benchnames) * len(frameworks)}"
        )
        for bench, fw in failed:
            print(f"{bench} / {fw}")
