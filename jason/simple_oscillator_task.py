from jason.rl_ctrnn import RL_CTRNN
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# Goal is to produce constantly (and maximally) changing neuron outputs
# A default reward function is provided which is reliant on external information tracking the NNs performance
class SimpleOscillatorTask():

    def __init__(self, duration, stepsize=0.01, stop_at_convergence=True, \
        reward_func = None, performance_func = None,\
        convergence_epsilon = 0.05, performance_update_rate=0.005, performance_bias=0.007  ):
        self.duration = duration                                    # how long to run at max
        self.stepsize = stepsize                                    # time to sim per iteration (i.e. 0.01)
        self.stop_at_convergence = stop_at_convergence              # stop simulating if the weights converge to a small enough value
        self.performance_update_rate = performance_update_rate      # how quickly the running average performance is updated
        self.performance_bias = performance_bias                    # small negative offset to encourage exploration
        self.convergence_epsilon = convergence_epsilon              # how small of a value to consider converged (not tested much)
        
        if reward_func == None:                           # uses performance to determine reward
            self.reward_func = self.default_reward_func
        else:
            self.reward_func = reward_func
        if performance_func == None:                      # determines how performance is measured
            self.performance_func = self.default_performance_func
        else:
            self.performance_func = performance_func
    
    # Default reward function
    # Assumes a performances and running_average_performances are recorded
    # Reward is calculated based upon the difference between the current performance and running average
    # Positive when performance is increasing, negative when decreasing
    # Can replace this function with a different one as desired
    def default_reward_func(self, nn):
        performance = self.performance_func(nn)
        running_average_performance = self.running_average_performances[self.step-1]
        # Current instantaneous performance vs. the current running average (NOT the previous instantaneous performance)
        return performance - running_average_performance

    # Default reward function
    # Assumes network has been advanced to the next step, but performance has not been measured
    # Assumes previous outputs, performances and running_average_performances are recorded
    def default_performance_func(self, nn):
        lastOutput = self.outputs[self.step-1]
        #                             difference in output                   penalty for not increasing
        performance = (np.sum( abs(nn.outputs - lastOutput ) ) ) / nn.size - self.performance_bias
        self.performances[self.step] = performance
        self.running_average_performances[self.step] = self.running_average_performances[self.step-1] * (1 - self.performance_update_rate ) \
            + self.performance_update_rate * performance
        return performance

    def simulate(self, nn, show_plots = False ):
        size = nn.size
        stepsize = self.stepsize

        #reset values each time - these are used in calculating
        time = np.arange(0.0, self.duration, stepsize)
        self.running_average_performances=np.zeros(len(time) )
        self.performances=np.zeros(len(time) )
        
        self.outputs=np.zeros((len(time),size))

        # Step is the current integer index from 0 up to len(time)
        self.step = 0
        nn.initializeState( np.zeros(size) )

        # Run simulation
        stop_step = -1        #if the experiment ends early due to convergence only plot up to that moment in time


        #track information to be used for plotting later
        plot_info={}
        
        if show_plots:
            # amplitude of fluctuation at each timestep of sim
            plot_info["amps"] = np.zeros(len(time) )
            # underlying true centers of weights of each synaptic connection (A->B) at each timestep of sim
            plot_info["weights"] = np.zeros((len(time),size,size))
            # the effective weights (including fluctuation) of each syn. conn. at each timestep of sim
            plot_info["flux_weights"] = np.zeros((len(time),size,size))
            #rewards received at each timestep
            plot_info["rewards"]=np.zeros(len(time) )

            if nn.bias_flux_mode:
                plot_info["biases"]=np.zeros((len(time),size) )
                plot_info["flux_biases"]=np.zeros((len(time),size) )

        for t in time:
            #avoid inflating reward in first step (going form zero to 0.5 is a huge change!)
            #This steps the reward at 0 (no change in output)
            if self.step == 0:
                self.outputs[-1] = nn.outputs

            nn.step(stepsize)

            #calculate reward by passing nn, info dict, and stepsize
            reward = self.reward_func(nn)

            #this should always happen to be consistent
            nn.update_weights_and_flux_amp_with_reward( reward )

            #record useful information
            self.outputs[self.step] = nn.outputs
            self.running_average_performances[self.step] = self.running_average_performances[self.step-1] * (1-self.performance_update_rate) + self.performance_update_rate * self.performances[self.step]
            
            if show_plots:
                plot_info["weights"][self.step] = nn.inner_weights
                plot_info["flux_weights"][self.step] = nn.calc_inner_weights_with_flux()
                plot_info["rewards"][self.step] = reward
                plot_info["amps"][self.step] = nn.flux_amp

                if nn.bias_flux_mode:
                    plot_info["biases"][self.step] = nn.biases
                    plot_info["flux_biases"][self.step] = nn.calc_bias_with_flux() if hasattr(nn, 'calc_bias_with_flux') else nn.biases
            
            self.step += 1
            
            #if stopping at convergence then when FluxAmp gets to zero stop
            if nn.flux_amp < self.convergence_epsilon and self.stop_at_convergence:
                stop_step = self.step
                break
        #END OF SIMULATION

        if show_plots:
            #switch the dimensions for easier plotting
            plot_info["outputs"] = self.outputs.transpose(1,0)
            plot_info["weights"] = plot_info["weights"].transpose(1,2,0)
            plot_info["flux_weights"] = plot_info["flux_weights"].transpose(1,2,0)

            if nn.bias_flux_mode:
                plot_info["biases"] = plot_info["biases"].transpose(1,0)
                plot_info["flux_biases"] = plot_info["flux_biases"].transpose(1,0)

            self.plot(plot_info, time, stop_step)
        #end if show_plots 
        
        converged = t < time[-1]

        plot_info["timed_passed"]=t

        return nn, plot_info, converged

    # stopStep provided because we only want to plot as long as the experiment ran
    def plot(self, plot_info, time, stop_step, reduce_plot_data=100):
        nnsize = len(plot_info["outputs"])
       
        #plot output of each neuron
        for i in range( nnsize ):
            #if reduce_plot_data == None:
            plt.plot(time[0:stop_step],plot_info["outputs"][i][0:stop_step],label=i, alpha=0.5 )
            #else:
            #    plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["outputs"][i][0:stop_step][0::reduce_plot_data],label=i, alpha=0.5 )
            
        plt.xlabel("Time")
        plt.ylabel("Output")
        plt.legend()
        plt.title("Neural activity")
        plt.show()
        if reduce_plot_data == None:
            plt.plot(time[0:stop_step],plot_info["rewards"][0:stop_step],label="instantaenous reward" )
        else:
            plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["rewards"][0:stop_step][0::reduce_plot_data],label="instantaenous reward" )
      
        #divided by (10 normally) so as to keep the vertical scale compressed (otherwise hard to see the other factors)
        highestAmp = np.max(plot_info["amps"] )
        plt.plot(time[0:stop_step],plot_info["amps"][0:stop_step]/highestAmp,label="flux amplitude/{:.2f} for scaling".format(highestAmp) )
        plt.plot(time[0:stop_step],self.running_average_performances[0:stop_step],label="running average (100 steps) rewards" )

        plt.xlabel("Time")
        plt.ylabel("Reward & Flux")
        plt.legend()
        plt.title("Reward")
        plt.show()

        # Plot Synaptic Weights over time
        for i in range(nnsize):
            for j in range(nnsize):
                if reduce_plot_data == None:
                    plt.plot(time[0:stop_step],plot_info["weights"][i][j][0:stop_step],label="{}->{}".format(i,j) )
                else:
                    plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["weights"][i][j][0:stop_step][0::reduce_plot_data],label="{}->{}".format(i,j) )


        plt.xlabel("Time")
        plt.ylabel("Weight Centers")
        plt.legend()
        plt.title("Synaptic Centers Strength over time")
        plt.show()

        # Plot Synaptic Weights over time
        for i in range(nnsize):
            for j in range(nnsize):
                if reduce_plot_data == None:
                    plt.plot(time[0:stop_step],plot_info["flux_weights"][i][j][0:stop_step],label="{}->{}".format(i,j) )
                else:
                    plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["flux_weights"][i][j][0:stop_step][0::reduce_plot_data],label="{}->{}".format(i,j) )

        plt.xlabel("Time")
        plt.ylabel("Fluctating Weights")
        plt.legend()
        plt.title("Fluctating Weights over time")
        plt.show()

        if "biases" in plot_info.keys():
            # Plot Biases over time
            for i in range(nnsize):
                plt.plot(time[0:stop_step],plot_info["biases"][i][0:stop_step],label="bias{}".format(i) )


            plt.xlabel("Time")
            plt.ylabel("Bias Centers")
            plt.legend()
            plt.title("Bias Centers  over time")
            plt.show()
        if "flux_biases" in plot_info.keys():
            # Plot Synaptic Weights over time
            for i in range(nnsize):
                plt.plot(time[0:stop_step],plot_info["flux_biases"][i][0:stop_step],label="bias{}".format(i) )


            plt.xlabel("Time")
            plt.ylabel("Fluctating Biases")
            plt.legend()
            plt.title("Fluctating Biases over time")
            plt.show()