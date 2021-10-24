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
    print("inside of xp_perturb_and_recovery.py")
    main_recovery()
    #main_perturb_AND_recover()

def main_recovery():
    ignore_transients=100  #20
    show_plots=False
    show_subplots=True
    seed=1
    size=2
    nnsize=size
    sol_seed=6
    init_flux=2
    performance_bias=0.01           #0.03
    performance_update_rate=0.001   #0.05   0.03
    flux_convergence= 1.5           #1.5
    learning_duration=1000

    directory=f"data/perturbed_networks/nnsize-{size}_sol-seed-{sol_seed}/seed{seed}/"
    filenames = os.listdir(directory)
    np.random.seed(seed)
    for filename in filenames:
        save_recover_data_directory=f"data/recovered_run_data/nnsize-{size}_sol-seed-{sol_seed}/seed{seed}/"
        save_recover_data_filename=f"{save_recover_data_directory}recover_{filename}.csv"
        load_filename=f"{directory}{filename}"
        ctrnn = CTRNN(size)
        ctrnn.load_json(load_filename)
        print(  filename  )
        recovered_fitness, timed_passed = run_recovery( ctrnn.get_normalized_parameters(), \
            init_flux=init_flux,\
            performance_bias=performance_bias, performance_update_rate=performance_update_rate, \
            nnsize=nnsize,learning_duration=learning_duration, flux_convergence=flux_convergence, \
            show_plots=show_plots, show_subplots=show_subplots, save_recover_data_filename=save_recover_data_filename, \
                ignore_transients=ignore_transients)
        print(recovered_fitness)
        #quit() #done after 1






def main_perturb_AND_recover():
    save_filename="jason/data/recovery_sweep_perturbed-fit_update-rate.csv"
    seeds=[0]  #range(10)
    jump_sizes=np.arange(0.1, 5, 0.1)
    #jump_sizes=[1,2,3]
    sol_seeds=[0]  #4 is best in 10nnsize
    nnsizes=[10,9,8,7,6,5,4,3,2]  #range(2,11)
    nnsizes=[2]
    learning_durations=[1000]
    #try progressively perturbing the network farther and farther in the same direction...
    line="jumpsize,sol_seed,seed,nnsize,orig_fit,perturbed_fit,perturbed_fit-div-orig_fit,orig_beer_fit,perturbed_beer_fit,perturbed_beer_fit-div-orig_beer_fit,perf_bias,perf_upd_rate,recovered_fitness,timed_passed,learning_duration,recover-div-orig,"
    
    if not os.path.exists(save_filename):
        print("File does not exist, writing to new file")
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


def run_recovery( norm_params, init_flux=1, nnsize=2, weight_range=16, bias_range=16,learning_duration=2000, performance_bias=0.005, \
    performance_update_rate=0.002, flux_convergence=1.0, show_plots=False, show_subplots=False, save_recover_data_filename=False,\
        ignore_transients=0 ):
    # Parameters RL-CTRNN specific
    init_flux_amp=init_flux
    #init_flux_amp=0.1
    max_flux_amp=8
    flux_period_min=4
    flux_period_max=8
    flux_conv_rate=flux_convergence
    learn_rate=1.0
    # could be tuned
    bias_init_flux_amp=0
    bias_max_flux_amp=0
    bias_flux_period_min=0
    bias_flux_period_max=0
    bias_flux_conv_rate=0

    save_nn_snapshots=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    ctrnn_save_directory="jason/data/ctrnn_snapshots_recovery/"



    convergence_epsilon=0.05
    stop_at_convergence=False
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
        convergence_epsilon=convergence_epsilon, performance_update_rate=performance_update_rate, performance_bias=performance_bias)

    save_data_filename=save_recover_data_filename

    nn, plot_info, converged = task.simulate(rl_nn, ignore_transients=ignore_transients, \
        show_plots=show_plots, show_subplots=show_subplots,  record_data=True, save_data_filename=save_data_filename,\
        save_nn_snapshots=save_nn_snapshots,  ctrnn_save_directory=ctrnn_save_directory)
    ctrnn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    ctrnn.set_normalized_parameters( nn.get_normalized_parameters() )
    recovered_fitness = fitness_maximize_output_change( ctrnn) 
    return recovered_fitness, plot_info
    


def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()

main()