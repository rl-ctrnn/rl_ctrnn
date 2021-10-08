import numpy as np
import numpy as np
import random
from jason.ctrnn import CTRNN
from jason.rl_ctrnn import RL_CTRNN
from jason.xp_perturb_solutions import run_perturb_experiment
from jason.mga import MicrobialGA
import matplotlib.pyplot as plt
import sys
import json
import os
from jason.simple_oscillator_task import SimpleOscillatorTask
from util.fitness_functions import fitness_maximize_output_change


def main():
    main_perturb_AND_recover()


def main_perturb_AND_recover():
    save_filename="jason/data/recovery_sweep_perturbed-fit_update-rate.csv"
    seeds=[0]  #range(10)
    jump_sizes=np.arange(0.1, 5, 0.1)
    #jump_sizes=[1,2,3]
    sol_seeds=[0]  #4 is best in 10nnsize
    nnsizes=[10,9,8,7,6,5,4,3,2]  #range(2,11)
    nnsizes=[2]
    learning_durations=[500]
    #try progressively perturbing the network farther and farther in the same direction...
    line="jumpsize,sol_seed,seed,nnsize,orig_fit,perturbed_fit,perturbed_fit-div-orig_fit,orig_beer_fit,perturbed_beer_fit,perturbed_beer_fit-div-orig_beer_fit,perf_bias,perf_upd_rate,recovered_fitness,timed_passed,learning_duration,recover-div-orig,"
    
    if not os.path.exists(save_filename):
        write_to_file( save_filename, line,'w' )
        print(line)

    performance_biases=[ 0.01,  0.015, 0.02, 0.05 ]
    performance_biases=[ 0.05 ]
    #performance_biases=[10, 5,2, 1, 0.5, 0.4, 0.3, 0.2, 0.1]
    performance_update_rates=[0.001, 0.025, 0.005, 0.01]

    performance_update_rates=[0.03]

    for learning_duration  in learning_durations:
        for nnsize in nnsizes:
            for sol_seed in sol_seeds:
                for jump_size in jump_sizes:
                    for performance_update_rate in performance_update_rates:
                        for performance_bias in performance_biases:
                            for seed in seeds:
                                line, norm_params, orig_fit = run_perturb_experiment( seed=seed, sol_seed=sol_seed, nnsize=nnsize, jump_size=jump_size)
                                
                                recovered_fitness, timed_passed = run_recovery( norm_params, performance_bias=performance_bias, performance_update_rate=performance_update_rate, nnsize=nnsize,learning_duration=learning_duration)
                                #print(recovered_fitness)
                                line2=f"{line}{performance_bias},{performance_update_rate},{recovered_fitness:.4f},{timed_passed:.1f},{learning_duration},{recovered_fitness/orig_fit:.4f}"
                                write_to_file( save_filename, line2,'a' )
                                print(line2)


def run_recovery( norm_params, nnsize=2, weight_range=16, bias_range=16,learning_duration=2000, performance_bias=0.005, \
    performance_update_rate=0.002 ):

    # Parameters RL-CTRNN specific
    init_flux_amp=1
    max_flux_amp=10
    flux_period_min=2
    flux_period_max=10
    flux_conv_rate=0.1
    learn_rate=1.0
    # could be tuned
    bias_init_flux_amp=1
    bias_max_flux_amp=10
    bias_flux_period_min=2
    bias_flux_period_max=10
    bias_flux_conv_rate=0.1

    #################### Work well for starting randomly for size 3nn
    # performance_bias=0.01
    # performance_update_rate=0.001
    #################### Work well for starting near solution
    # performance_bias=0.01
    # performance_update_rate=0.005
    # performance_bias=0.005
    # performance_update_rate=0.002



    convergence_epsilon=0.05
    stop_at_convergence=True

    gaussian_mode=True
    # All Tasks
    stepsize=0.01

    tc_min=1
    tc_max=1

    rl_nn = RL_CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max,\
        init_flux_amp=init_flux_amp, max_flux_amp=max_flux_amp, flux_period_min=flux_period_min, flux_period_max=flux_period_max, flux_conv_rate=flux_conv_rate, learn_rate=learn_rate,\
        gaussian_mode=gaussian_mode, \
        bias_init_flux_amp=bias_init_flux_amp, bias_max_flux_amp=bias_max_flux_amp, \
        bias_flux_period_min=bias_flux_period_min,bias_flux_period_max=bias_flux_period_max,\
        bias_flux_conv_rate=bias_flux_conv_rate)

    rl_nn.set_normalized_parameters(norm_params)

    task = SimpleOscillatorTask( learning_duration, stepsize, stop_at_convergence, \
        convergence_epsilon=convergence_epsilon, performance_update_rate=performance_update_rate, performance_bias=performance_bias )

    nn, plot_info, converged = task.simulate(rl_nn, show_plots=False)

    ctrnn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    ctrnn.set_normalized_parameters( nn.get_normalized_parameters() )

    recovered_fitness = fitness_maximize_output_change( ctrnn) 

    return recovered_fitness, plot_info["timed_passed"]
    


def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()

main()