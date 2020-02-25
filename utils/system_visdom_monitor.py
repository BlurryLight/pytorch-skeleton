#! /usr/bin/env python
#! coding:utf-8

import visdom
import time
import numpy as np
import argparse
import pynvml
import psutil


def gpu_info_init():
    """
    input:None
    return: handle
    only support one card
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle


def get_cpu_mem():
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory()[2]
    return cpu_percent, mem_percent


def get_gpu_info(handle):
    # """
    # input: handle of GPU
    # output:cuda version,gpu name,total memory,used_memory,free memory, gpu_util_rate
    # """
    # https://docs.nvidia.com/deploy/nvml-api/
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total
    free_memory = info.free
    used_memory = info.used
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util_rate = utilization.gpu
    return pynvml.nvmlSystemGetDriverVersion(), pynvml.nvmlDeviceGetName(handle), total_memory, free_memory, used_memory, gpu_util_rate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', '-s', type=str, default='http://localhost',
                        help='visdom server url/ip address')
    parser.add_argument('--port', '-p', type=int, default=8097,
                        help='visdom server port')
    parser.add_argument('--base_url', '-b', type=str, default='/',
                        help='Visdom Base url')
    parser.add_argument('--env_name', '-n', type=str, default='env' + str(int(time.time()//60)),
                        help='Visdom env name,default is env_time_from_epoch')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    viz = visdom.Visdom(
        port=args.port,
        server=args.server,
        base_url=args.base_url,
        # username=...
        # password=..
        use_incoming_socket=False,
        env=args.env_name
    )
    gpu_handle = gpu_info_init()

    origin = np.zeros((1, 3))
    gpu_memory_win = viz.line(
        env=args.env_name,
        X=origin,
        Y=origin,
        opts=dict(
            showlegend=True,
            xlabel='time',
            ylabel='gpu memory',
            title='gpu memory',
            ytickmin=0,
            legend=['total', 'free', 'used']
        ),
    )

    origin = np.zeros((1, 4))
    percent_win = viz.line(
        env=args.env_name,
        X=origin,
        Y=origin,
        opts=dict(
            showlegend=True,
            xlabel='time',
            ylabel='percent',
            title='cpu/mem/gpu percent',
            ytickmin=0,
            legend=['cpu', 'mem', 'gpu_mem', 'gpu_usage']
        ),
    )
    index = 1
    while True:
        # tm : totol memory
        # fm : free memory
        # um : used memory
        _, _, tm, fm, um, gpu_util_percent = get_gpu_info(gpu_handle)
        tm, fm, um = tm//(1024*1024), fm//(1024*1024), um//(1024*1024)
        cpu_p, mem_p = get_cpu_mem()
        gpu_p = (um / tm) * 100

        # X,Y shape must match
        viz.line(
            env=args.env_name,
            X=np.full((1, 3), index, dtype=int),
            Y=np.array([[tm, fm, um]]),
            win=gpu_memory_win,
            update='append'
        )

        viz.line(
            env=args.env_name,
            X=np.full((1, 4), index, dtype=int),
            Y=np.array([[cpu_p, mem_p, gpu_p, gpu_util_percent]]),
            win=percent_win,
            update='append'
        )
        index = index + 3
        time.sleep(3)
