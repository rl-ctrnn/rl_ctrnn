import numpy as np
import random
from jason.ctrnn import CTRNN
from jason.rl_ctrnn import RL_CTRNN
from jason.xp_perturb_and_recover import run_recovery
import matplotlib.pyplot as plt
import sys
import json
import os
from jason.simple_oscillator_task import SimpleOscillatorTask
from util.fitness_functions import fitness_maximize_output_change


def main():
    repeatedly_train_weights_from_starting_file("starting_nn.json")

def repeatedly_train_weights_from_starting_file(filename):

    #data to record to files
    record_array=["performances", "running_average_performances", "outputs", "weights", "flux_weights", "amps"]

    #how frequently to record data
    record_every_n_steps=100

    #how long to ignore the intial transients
    ignore_transients=100  #20

    show_plots=False
    show_subplots=False
    size=2
    nnsize=size


    init_flux=4
    performance_bias=0.05           #0.03
    performance_update_rate=0.001   #0.05   0.03
    flux_convergence= 1.5           #1.5
    learning_duration=1000  #in seconds

    running_window_mode=True
    running_window_size=2000   # 2000 = 20 seconds ~= 0.001

    directory=f"cooper/figure2/"

    #how many times to run the simulation
    seeds = range(0,100)
    
    for seed in seeds:
        np.random.seed(seed)
        print(seed)
        save_recover_data_directory=f"cooper/figure2/rl_data/"
        save_recover_data_filename=f"{save_recover_data_directory}recover_seed-{seed}.csv"
        load_filename=f"{directory}{filename}"
        ctrnn = CTRNN(size)
        ctrnn.load_json(load_filename)

        recovered_fitness, plot_info = run_recovery( ctrnn.get_normalized_parameters(), \
            init_flux=init_flux,running_window_mode=running_window_mode, running_window_size=running_window_size, \
            performance_bias=performance_bias, performance_update_rate=performance_update_rate, \
            nnsize=nnsize,learning_duration=learning_duration, flux_convergence=flux_convergence, \
            show_plots=show_plots, show_subplots=show_subplots, save_recover_data_filename=save_recover_data_filename, \
            ignore_transients=ignore_transients,
            record_array=record_array,
            record_every_n_steps=record_every_n_steps)
        print( f"{recovered_fitness}     ..    {(plot_info['running_average_performances'][-1]+performance_bias)*100}" )


if __name__ == "__main__":
    main()