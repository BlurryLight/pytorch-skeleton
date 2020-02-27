#! /usr/bin/env python
#! coding:utf-8
import visdom
import numpy as np


def visdom_init(viz_server, viz_port, viz_base_url, viz_env_name):
    viz = visdom.Visdom(port=viz_port, base_url=viz_base_url,
                        env=viz_env_name, server=viz_server, use_incoming_socket=False)
    assert viz.check_connection(), "visdom server isn't running!"
    train_loss_handle = viz.line(env=viz_env_name,
                                 X=np.array([-1]), Y=np.array([0]),
                                 opts=dict(
                                     xlabel='epoch',
                                     ylabel='loss',
                                     title='train loss curve',
                                     xtickmin=0,
                                     ytickmin=0
                                 ),
                                 )
    origin2d = np.array([[-1, -1]])
    acc_handle = viz.line(env=viz_env_name,
                          X=origin2d, Y=np.array([[0, 0]]),
                          opts=dict(
                              showlegend=True,
                              legend=['train', 'test'],
                              xlabel='epoch',
                              ylabel='acc',
                              title='train acc curve',
                              xtickmin=0,
                              ytickmin=0,
                              ytickmax=100
                          ),
                          )
    return viz, train_loss_handle, acc_handle
