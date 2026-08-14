#!/usr/bin/python3
#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys

# Cap the CPU math thread pools before onnxruntime is imported (via
# edge_ai_class -> edgeai_dl_inferer), since the pool size is fixed at import.
#
# Any model layer that TIDL cannot offload runs on the ARM through the CPU EP.
# For the UFLDv2 cls.3 Gemm, measurements on AM67A showed the extra threads buy
# nothing -- 17.71 ms with 4 threads vs 17.69 ms with 1 -- while they contend
# with the post-process thread and inflate inference by 10.3 ms instead of
# 6.5 ms. Set EDGEAI_CPU_THREADS to override.
_cpu_threads = int(os.environ.get("EDGEAI_CPU_THREADS", "2"))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(_cpu_threads))


def _limit_ort_cpu_threads(n_threads):
    """
    Bound onnxruntime's intra-op thread pool and stop it spin-waiting.

    Layers TIDL cannot offload run on the ARM through the CPU EP, and ORT sizes
    its pool to the core count and busy-waits between calls. Measured on AM67A
    with the UFLDv2 cls.3 Gemm:

        default            17.71 ms/inf, 59.73 ms CPU -> 3.37 cores busy
        intra_op=2, nospin 19.56 ms/inf, 13.43 ms CPU -> 0.69 cores busy

    So the pool was spending ~2.7 extra cores to buy 1.85 ms of latency, which
    starved the post-process thread. The env vars above do NOT affect this --
    ORT's pool is configured through SessionOptions, so patch the constructor
    that edgeai_dl_inferer calls. Set EDGEAI_CPU_THREADS=0 to leave it alone.
    """
    if n_threads <= 0:
        return
    try:
        import onnxruntime as _ort
    except ImportError:
        return

    _real_init = _ort.InferenceSession

    def _capped(*args, **kwargs):
        so = kwargs.get("sess_options")
        if so is None:
            so = _ort.SessionOptions()
            kwargs["sess_options"] = so
        so.intra_op_num_threads = n_threads
        so.inter_op_num_threads = 1
        try:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass
        return _real_init(*args, **kwargs)

    _ort.InferenceSession = _capped


_limit_ort_cpu_threads(_cpu_threads)

import yaml

from edge_ai_class import EdgeAIDemo
import utils


def main(sys_argv):
    args = utils.get_cmdline_args(sys_argv)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    try:
        demo = EdgeAIDemo(config)
        demo.start()

        if args.verbose:
            utils.print_stdout = True

        if not args.no_curses:
            utils.enable_curses_reports(demo.title)

        demo.wait_for_exit()
    except KeyboardInterrupt:
        demo.stop()
    finally:
        pass

    utils.disable_curses_reports()

    del demo


if __name__ == "__main__":
    main(sys.argv)
