import numpy as np
import numpy as np
import random
from jason.ctrnn import CTRNN
from jason.rl_ctrnn import RL_CTRNN
from jason.mga import MicrobialGA
import matplotlib.pyplot as plt
import sys
import json
import os
from jason.simple_oscillator_task import SimpleOscillatorTask
from util.fitness_functions import fitness_maximize_output_change

def main():
    main_perturb_only()

def main_perturb_only():
    save_filename="jason/data/perturbation_trajectory_new_20x1000_evo.csv"
    seeds=range(10)
    seeds=[0]
    jump_sizes=np.arange(0.1, 5, 0.1)
    sol_seeds=[0]
    nnsizes=[2,10]  #range(2,11)
    #try progressively perturbing the network farther and farther in the same direction...
    line="jumpsize,sol_seed,seed,nnsize,orig_fit,perturbed_fit,perturbed_fit-div-orig_fit,orig_beer_fit,perturbed_beer_fit,perturbed_beer_fit-div-orig_beer_fit,"
    if not os.path.exists(save_filename):
        write_to_file( save_filename, line,'w' )
        print(line)
    for nnsize in nnsizes:
        for sol_seed in sol_seeds:
            for seed in seeds:
                for jump_size in jump_sizes:
                    line, norm_params, orig_fit = run_perturb_experiment( seed=seed, sol_seed=sol_seed, nnsize=nnsize, jump_size=jump_size)
                    write_to_file( save_filename, line,'a' )
                    print(line)


def run_perturb_experiment( seed=0, sol_seed=1,jump_size=2, nnsize=2, weight_range=16, bias_range=16 ):
    """ Load highly evolved solution from file and then perturb weights in increasing magnitude in a specific direction."""
    np.random.seed(seed)
    # parameters for the preloaded file
    tc_min=1
    tc_max=1
    # Non-Learning Task
    non_learning_duration=10
    #this is true of both the rl_ctrnn and the ctrnn
    genesize = nnsize * nnsize + 2 * nnsize

    # def ctrnn_genotype_ff(genotype):
    #     ctrnn = CTRNN(nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    #     ctrnn.set_normalized_parameters( genotype )
    #     return non_learning_ff_ctrnn(ctrnn,duration=non_learning_duration)

    ##############################################################################
    #Load a previously evolved example ctrnn
    best_nn = CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )
    best_evolved_filename=f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{nnsize}_seed-{sol_seed}.json"
    best_nn.load_json( best_evolved_filename )

    new_ctrnn = CTRNN( nnsize, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max )

    #faster to simulate and easier to evolve
    orig_fit=fitness_maximize_output_change(best_nn)
    
    #imitating Beer's non-transient version of task
    orig_beer_fit=fitness_maximize_output_change(best_nn,init_duration=250,test_duration=50)


    #Pick random direction and perturb network progressively
    random_vector = np.random.uniform(low=-1,high=1,size=(nnsize,nnsize) )
    #random_normalized_noise = weight_range * weight_range * random_vector / np.sqrt( np.sum(random_vector**2) )
    random_normalized_noise = 2 * 2 * random_vector / np.sqrt( np.sum(random_vector**2) )

   
    new_ctrnn.load_json( best_evolved_filename )
    random_noise_with_jump_size = random_normalized_noise * jump_size
    new_ctrnn.inner_weights = new_ctrnn.inner_weights + random_noise_with_jump_size

    #vector difference
    diff_vec = (best_nn.inner_weights - new_ctrnn.inner_weights )            
    #Cartesian distance
    before_clip = np.sqrt( np.sum(diff_vec**2) ) 
    #keep within limits
    new_ctrnn.inner_weights = np.clip(new_ctrnn.inner_weights,-weight_range,weight_range)
    #new vector difference
    diff_vec = (best_nn.inner_weights - new_ctrnn.inner_weights )  
    #new Cartesian distance
    after_clip = np.sqrt( np.sum(diff_vec**2) )   

    perturbed_fit=fitness_maximize_output_change(new_ctrnn)
    perturbed_beer_fit=fitness_maximize_output_change(new_ctrnn,init_duration=250,test_duration=50)

    norm_params = new_ctrnn.get_normalized_parameters()

    # rl_nn.set_normalized_parameters(params)
    # fit = rl_ctrnn_ff( rl_nn, show_plots=show_plots )
    #jumpsize,,seed,,orig_fit
    line=f"{jump_size:.1f},{sol_seed},{seed},{nnsize},{orig_fit:.4f},{perturbed_fit:.4f},{perturbed_fit/orig_fit:.4f},{orig_beer_fit:.4f},{perturbed_beer_fit:.4f},{perturbed_beer_fit/orig_beer_fit:.4f},"
    return line, norm_params, orig_fit




def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()

main()