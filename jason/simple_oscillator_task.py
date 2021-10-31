from matplotlib import colors
from jason.rl_ctrnn import RL_CTRNN
from jason.ctrnn import CTRNN
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from vis.colorline import colorline
import matplotlib.path as mpath


# Goal is to produce constantly (and maximally) changing neuron outputs
# A default reward function is provided which is reliant on external information tracking the NNs performance
class SimpleOscillatorTask():

    def __init__(self, duration, stepsize=0.01, stop_at_convergence=True, \
        reward_func = None, performance_func = None,\
        convergence_epsilon = 0.05, performance_update_rate=0.005, performance_bias=0.007, \
            running_window_mode=False, running_window_size=2000  ):
        self.duration = duration                                    # how long to run at max
        self.stepsize = stepsize                                    # time to sim per iteration (i.e. 0.01)
        self.stop_at_convergence = stop_at_convergence              # stop simulating if the weights converge to a small enough value
        self.running_window_mode = running_window_mode
        if self.running_window_mode:
            self.running_window_size = running_window_size
            self.sliding_window = np.zeros(self.running_window_size)
            #self.sliding_window = np.ones(self.running_window_size)*-1*performance_bias
        else:
            self.performance_update_rate = performance_update_rate  # how quickly the running average performance is updated
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

        if self.running_window_mode:
            #rotate everything forward
            self.sliding_window = np.roll(self.sliding_window, 1)
            #replace oldest value (which just rolled to the front)
            self.sliding_window[0] = performance
            #current running average
            self.running_average_performances[self.step] = np.mean(self.sliding_window)

        else:
            self.running_average_performances[self.step] = self.running_average_performances[self.step-1] * (1 - self.performance_update_rate ) \
                + self.performance_update_rate * performance
        return performance

    def simulate(self, nn, ignore_transients=50, show_plots = False, show_subplots=False, record_data=True, save_data_filename=None, save_nn_snapshots=None, ctrnn_save_directory=None):
        if show_plots or show_subplots:
            record_data=True #cannot show unless recorded

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
        
        if record_data:
            # time passed to use for plotting
            plot_info["time"] = time
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
            

            if not save_nn_snapshots == None:
                if t in save_nn_snapshots:
                    nn.save_json( f"{ctrnn_save_directory}time-{t}.json")


            nn.step(stepsize)

            reward = self.reward_func(nn)

            if t < ignore_transients:
                reward = 0
            
            #this should always happen to be consistent
            nn.update_weights_and_flux_amp_with_reward( reward )

            #record useful information
            self.outputs[self.step] = nn.outputs
            if self.running_window_mode:
                #rotate everything forward
                self.sliding_window = np.roll(self.sliding_window, 1)
                #replace oldest value (which just rolled to the front)
                self.sliding_window[0] = self.performances[self.step]
                #current running average
                self.running_average_performances[self.step] = np.mean(self.sliding_window)
            else:
                self.running_average_performances[self.step] = self.running_average_performances[self.step-1] * (1-self.performance_update_rate) + self.performance_update_rate * self.performances[self.step]
            
            if record_data:
                plot_info["weights"][self.step] = nn.inner_weights
                #have to reverse the transpose to get the same coordinates (gets multipled)
                plot_info["flux_weights"][self.step] = nn.calc_inner_weights_with_flux().T
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

        if record_data:
            #switch the dimensions for easier plotting
            plot_info["outputs"] = self.outputs.transpose(1,0)
            plot_info["weights"] = plot_info["weights"].transpose(1,2,0)
            plot_info["flux_weights"] = plot_info["flux_weights"].transpose(1,2,0)

            if nn.bias_flux_mode:
                plot_info["biases"] = plot_info["biases"].transpose(1,0)
                plot_info["flux_biases"] = plot_info["flux_biases"].transpose(1,0)

            plot_info["running_average_performances"] = self.running_average_performances
            plot_info["performances"] = self.performances

            if show_plots:
                self.plot(plot_info, time, stop_step)
            if show_subplots:
                self.subplots(plot_info, time, stop_step)
        #end if record_data 
        
        converged = t < time[-1]

        plot_info["time_passed"]=t
        #redundant but useful for plotting later
        plot_info["time"]=time

        if not save_data_filename == None:
            self.save_plot_info( plot_info, nn.size, nn.bias_flux_mode, save_data_filename )

        return nn, plot_info, converged
    
    def save_plot_info(self, plot_info, nn_size, nn_bias_flux_mode, filename ):
        keys=plot_info.keys()
        header="time,amps,rewards,performances,running_average_performances,"
        for i in range(nn_size):
            header+=f"output{i},"
        for i in range(nn_size):
            for j in range(nn_size):
                header+=f"w{i}_{j},"
        for i in range(nn_size):
            for j in range(nn_size):
                header+=f"flux_w{i}_{j},"
        if nn_bias_flux_mode:
            for i in range(nn_size):
                header+=f"bias{i},"
            for i in range(nn_size):
                header+=f"flux_bias{i},"
        # for i in range(nn_size):
        #     header+=f"tc{i},"
       
        write_to_file(filename, header, 'w' )
        for t in range(len(plot_info["time"])):
            
            line=f'{plot_info["time"][t]},{plot_info["amps"][t]},{plot_info["rewards"][t]},{plot_info["performances"][t]},{plot_info["running_average_performances"][t]},'
            for i in range(nn_size):
                line+=f'{plot_info["outputs"][i][t]},'
            
            for i in range(nn_size):
                for j in range(nn_size):
                    line+=f'{plot_info["weights"][i][j][t]},'
            for i in range(nn_size):
                for j in range(nn_size):
                    line+=f'{plot_info["flux_weights"][i][j][t]},'
            if nn_bias_flux_mode:
                for i in range(nn_size):
                    line+=f'{plot_info["biases"][i][t]},'
                for i in range(nn_size):
                    line+=f'{plot_info["flux_biases"][i][t]},'
            # for i in range(nn_size):
            #     line+=f'{plot_info["outputs"][i][t]},'

            write_to_file(filename, line, 'a' )



    # stopStep provided because we only want to plot as long as the experiment ran
    def plot(self, plot_info, time, stop_step, reduce_plot_data=100):
        nnsize = len(plot_info["outputs"])


        plt.xlabel("Time")
        plt.ylabel("Performance")
        if reduce_plot_data == None:
            plt.plot(time[0:stop_step],plot_info["rewards"][0:stop_step],label="instantaenous reward" )
            plt.plot(time[0:stop_step],plot_info["performances"][0:stop_step],label="instantaenous performance" )
            plt.plot(time[0:stop_step],plot_info["running_average_performances"][0:stop_step],label="running average performance" )
            highestAmp = np.max(plot_info["amps"] )
            plt.plot(time[0:stop_step],plot_info["amps"][0:stop_step]/highestAmp/100,label="flux amplitude/{:.2f} for scaling".format(highestAmp*100) )
        else:
            plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["rewards"][0:stop_step][0::reduce_plot_data],label="instantaenous reward" )
            plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["performances"][0:stop_step][0::reduce_plot_data],label="instantaenous performance" )
            plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["running_average_performances"][0:stop_step][0::reduce_plot_data],label="running average performance" )
            highestAmp = np.max(plot_info["amps"] )
            plt.plot(time[0:stop_step][0::reduce_plot_data],plot_info["amps"][0:stop_step][0::reduce_plot_data]/(highestAmp*100),label="flux amplitude/{:.2f} for scaling".format(highestAmp*100) )



        plt.legend()
        plt.title("Fluctuation Amplitude, Instantaneous and Running Average Performance over Time")
        plt.show()




       
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
        plt.ylabel("Fluctuating Weights")
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
    
    
    def subplots(self, plot_info, time, stop_step, reduce_plot_data=10):


        reduce_plot_data=100

        start_time=100
        start_step=start_time*100
        nnsize = len(plot_info["outputs"])

        rows=3
        cols=2
        if "biases" in plot_info.keys():
            cols+=1

        fig, axs = plt.subplots(rows, cols)
        subplot_index=1


##############################
        ax1 = plt.subplot(rows, cols, subplot_index)

        #plt.xlabel("Time")
        #plt.ylabel("Performance")
        # if reduce_plot_data == None:
        #     ax2=ax1.twinx()
        #     ax1.plot(time[start_step:stop_step],plot_info["amps"][start_step:stop_step],label="flux amplitude", color='r' )
        #     #ax1.set_ylim(0,8)
        #     ax2.plot(time[start_step:stop_step],plot_info["performances"][start_step:stop_step],label="instantaneous", color='c' )
        #     ax2.plot(time[start_step:stop_step],plot_info["running_average_performances"][start_step:stop_step],label="running average", color='b' )


        # else:
        ax2=ax1.twinx()
        ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["amps"][start_step:stop_step:reduce_plot_data],label="flux amplitude",color='r' )
        #ax1.set_ylim(0,8)
        ax2.plot(time[start_step:stop_step:reduce_plot_data],plot_info["performances"][start_step:stop_step:reduce_plot_data],label="instantaneous", color='c' )
        ax2.plot(time[start_step:stop_step:reduce_plot_data],plot_info["running_average_performances"][start_step:stop_step:reduce_plot_data],label="running average", color='b' )

        ax1.set_ylabel("Fluctuation Size",color='r',labelpad=-40)
        ax2.set_ylabel("Performance", color='b',labelpad=-60)

        cmap = plt.cm.turbo
        custom_lines = [Line2D([0], [0], color='r', lw=4),
                        Line2D([0], [0], color='c', lw=4),
                        Line2D([0], [0], color='b', lw=4)]
        legend_array=['fluctuation', 'instantaneous', 'running average']

        ax1.legend(custom_lines, legend_array, loc='lower right')
        
        ax1.set_title("Fluctuation and Performance over Time")


###################################
        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)


        # Plot Synaptic Weights over time
        for i in range(nnsize):
            for j in range(nnsize):
                # if reduce_plot_data == None:
                #     ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["weights"][i][j][start_step:stop_step],label="{}->{}".format(i,j) )
                # else:
                ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data],label="{}->{}".format(i,j) )


        #plt.xlabel("Time")
        #plt.ylabel("Weight Centers")
        ax1.legend(loc='lower right')
        ax1.set_title("Synaptic Weight Centers over time")
        #plt.show()
###########################
        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)

       
        #plot output of each neuron
        for i in range( nnsize ):
            #if reduce_plot_data == None:
            ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["outputs"][i][start_step:stop_step:reduce_plot_data],label=i, alpha=0.5 )
            #else:
            #    plt.plot(time[start_step:stop_step][start_step::reduce_plot_data],plot_info["outputs"][i][start_step:stop_step][start_step::reduce_plot_data],label=i, alpha=0.5 )
            
        #plt.xlabel("Time")
        #plt.ylabel("Output")
        ax1.legend(loc='lower right')
        ax1.set_title("Neural Outputs")      
###################################

        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)

        # Plot Synaptic Weights over time
        for i in range(nnsize):
            for j in range(nnsize):
                if reduce_plot_data == None:
                    ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data],label="{}->{}".format(i,j) )
                else:
                    ax1.plot(time[start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data],label="{}->{}".format(i,j) )
        
    

        #plt.xlabel("Time")
        #plt.ylabel("Fluctating Weights")
        ax1.legend(loc='lower right')
        ax1.set_title("Fluctuating Weights over time")
###################################
        subplot_index+=1
        ax1 = plt.subplot(rows, cols, subplot_index)

        cmapname = 'viridis'
        cmap=plt.get_cmap(cmapname)
        flux_linewidth=1
        center_linewidth=2
        flux_alpha=0.25
        center_alpha=1
        endpoint_size=5

        start_color=cmap(0.0)
        stop_color=cmap(0.99)

        r,g,b = cmap(0.5)[0], cmap(0.5)[1], cmap(0.5)[2]

        custom_lines = [Line2D([0], [0], color=cmap(0), lw=4),
                        Line2D([0], [0], color=(r,g,b,center_alpha) , lw=4),
                        Line2D([0], [0], color=(r,g,b,flux_alpha), lw=4),
                        Line2D([0], [0], color=cmap(0.99), lw=4)]
        legend_array=['start', 'weight center', 'fluctuating weight', 'stop']

        i=0
        j=1

        ax1.scatter( plot_info["weights"][i][i][start_step], plot_info["weights"][j][j][start_step],label="start", color=start_color, linewidths=endpoint_size )
        ax1.scatter( plot_info["weights"][i][i][stop_step], plot_info["weights"][j][j][stop_step],label="end", color=stop_color, linewidths=endpoint_size )
        #ax1.scatter( plot_info["weights"][i][i][int(stop_step/2)], plot_info["weights"][j][j][int(stop_step/2)],label="middle", color='orange' )
        ax1.set_xlabel("Weight {}->{}".format(i,i))
        ax1.set_ylabel("Weight {}->{}".format(j,j))

        


        wii=plot_info["weights"][i][i][start_step:stop_step:reduce_plot_data]
        wjj=plot_info["weights"][j][j][start_step:stop_step:reduce_plot_data]
        fwii=plot_info["flux_weights"][i][i][start_step:stop_step:reduce_plot_data]
        fwjj=plot_info["flux_weights"][j][j][start_step:stop_step:reduce_plot_data]
        ax1.set_ylim(np.min(fwjj), np.max(fwjj))
        ax1.set_xlim(np.min(fwii), np.max(fwii))
        lcfw = create_colored_lc( fwii, fwjj, alpha=flux_alpha, linewidth=flux_linewidth,cmapname=cmapname)
        ax1.add_collection(lcfw)
        lc = create_colored_lc( wii, wjj, alpha=center_alpha, linewidth=center_linewidth,cmapname=cmapname)
        ax1.add_collection(lc)



        # ax1.plot(plot_info["weights"][i][i][start_step:stop_step:reduce_plot_data],plot_info["weights"][j][j][start_step:stop_step:reduce_plot_data],color=[0,1,0,0.5] )
        # ax1.plot(plot_info["flux_weights"][i][i][start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][j][j][start_step:stop_step:reduce_plot_data], color=[0,1,0,0.2] )
        ax1.set_title("Fluctuating Weights over time")
        ax1.legend(custom_lines, legend_array, loc='lower right')

        subplot_index+=1

        ax1 = plt.subplot(rows, cols, subplot_index)

        ax1.scatter( plot_info["weights"][i][j][start_step], plot_info["weights"][j][i][start_step],label="start", color=start_color )
        ax1.scatter( plot_info["weights"][i][j][stop_step], plot_info["weights"][j][i][stop_step],label="end", color=stop_color )
        #ax1.scatter( plot_info["weights"][i][j][int(stop_step/2)], plot_info["weights"][j][i][int(stop_step/2)],label="middle", color='orange' )

        #ax1.plot(plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data],plot_info["weights"][j][i][start_step:stop_step:reduce_plot_data],color=[0,1,0,0.5] )
        #ax1.plot(plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data],plot_info["flux_weights"][j][i][start_step:stop_step:reduce_plot_data], color=[0,1,0,0.2] )
        
        wij=plot_info["weights"][i][j][start_step:stop_step:reduce_plot_data]
        wji=plot_info["weights"][j][i][start_step:stop_step:reduce_plot_data]
        fwij=plot_info["flux_weights"][i][j][start_step:stop_step:reduce_plot_data]
        fwji=plot_info["flux_weights"][j][i][start_step:stop_step:reduce_plot_data]
        ax1.set_ylim(np.min(fwji), np.max(fwji))
        ax1.set_xlim(np.min(fwij), np.max(fwij))
        lcfw = create_colored_lc( fwij, fwji, alpha=flux_alpha, linewidth=flux_linewidth,cmapname=cmapname)
        ax1.add_collection(lcfw)
        lc = create_colored_lc( wij, wji, alpha=center_alpha, linewidth=center_linewidth,cmapname=cmapname)
        ax1.add_collection(lc)
        
        
        
        ax1.set_title("Fluctuating Weights over time")
        ax1.set_xlabel("Weight {}->{}".format(i,j))
        ax1.set_ylabel("Weight {}->{}".format(j,i))


        
        ax1.legend(custom_lines, legend_array, loc='lower right')

        

###################################
        if "biases" in plot_info.keys():
            subplot_index+=1
            ax1 = plt.subplot(rows, cols, subplot_index)


        if "biases" in plot_info.keys():
            # Plot Biases over time
            for i in range(nnsize):
                plt.plot(time[start_step:stop_step:reduce_plot_data],plot_info["biases"][i][start_step:stop_step:reduce_plot_data],label="bias{}".format(i) )


            plt.xlabel("Time")
            plt.ylabel("Bias Centers")
            plt.legend()
            plt.title("Bias Centers  over time")
            plt.show()
        if "flux_biases" in plot_info.keys():
            subplot_index+=1
            ax1 = plt.subplot(rows, cols, subplot_index)
            # Plot Synaptic Weights over time
            for i in range(nnsize):
                plt.plot(time[start_step:stop_step:reduce_plot_data],plot_info["flux_biases"][i][start_step:stop_step:reduce_plot_data],label="bias{}".format(i) )


            plt.xlabel("Time")
            plt.ylabel("Fluctating Biases")
            plt.legend()
            plt.title("Fluctating Biases over time")
            plt.show()


        plt.show()

def create_colored_lc(x,y, alpha=1.0, linewidth=2, cmapname='jet'):
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    lc = colorline(x, y, z, cmap=plt.get_cmap(cmapname), linewidth=linewidth, alpha=alpha)
    return lc

def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()
