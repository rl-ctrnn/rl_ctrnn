import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def alpha_adj(color, alpha=0.25):
    """
    Adjust alpha of color.
    """
    return [color[0], color[1], color[2], alpha]

def improvement_adj(init_fit, final_fit):
    """
    Calculate improvement of y over x.
    """
    #return x
    # if init_fit > final_fit:
    #     print("it got worse")
    

    maxxed= max(0,  0.5 + (final_fit - init_fit)  )
    # minned = min(1, max(0,  0.5 + (final_fit - init_fit)  ))
    # if not maxxed == minned:
    #     print(maxxed, minned)


    return maxxed
    #return min(1, max(0,  0.5 + (final_fit - init_fit)  ))


#DEMO
# pairs=[[0.5, 0.4],
#         [0.7, 0.05],
#         [0.7, 0.1],
#         [0.7, 0.2],
#         [0.7, 0.3],
#         [0.7, 0.2],
#         [0.4, 0.5],
#         [0.4, 0.7],
#         [0.05, 0.7],
#         [0.1, 0.7],
#         [0.2, 0.7],
#         [0.3, 0.7],
#         [0.2, 0.7],
#         [0.4, 0.5]]

# for init_fit, final_fit in pairs:


#     improvement=0.5 + (final_fit-init_fit)
#     cmap = plt.cm.PiYG

#     plt.scatter( init_fit, final_fit, color=alpha_adj(cmap(improvement) ) )

# x_min = plt.xlim()
# y_min = plt.ylim()
# plt.xlim(0,x_min[1])
# plt.ylim(0,y_min[1])

# plt.plot( [0,100], [0,100], color=[.5,.5,.5,0.25] )


# plt.show()






#####################################################

legend_array=['much worse', 'worse', 'no change', 'better', 'much better']

#     500, 2000
# learning_duration=5000
# initflux=2
init_flux=1
learning_duration=1000
prefix="v1_12x12x4"
save_dat_filename=f"{prefix}_fig2_{learning_duration/1000}k_initflux-{init_flux}.dat"




data = np.genfromtxt(save_dat_filename,delimiter=",", dtype=float)

cmap = plt.cm.gnuplot
cmap = plt.cm.PiYG
cmap = plt.cm.turbo
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.25), lw=4), 
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(.75), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]


first_time_index=13

threshold=0.5
success=0
for index in range(len(data)):
#     #init_fit,final_fit,init_est_dist,final_est_dist
#     init_fit=data[index][0]
    final_fit=data[index][1]
    #init_dist=data[index][2]
    #final_dist=data[index][3]

 
#     improvement=improvement_adj(init_fit, final_fit)
#     start=0
    if final_fit>threshold:
        alpha=1
        success+=1
#     else:
#         alpha=0.1    
    # plt.plot( range(learning_duration-start), data[index][start+first_time_index:], color=alpha_adj(cmap(improvement) ) )
    #plt.plot( range(500-start), data[index][start+5:], color=[improvement,0,0,alpha] )
# plt.legend(custom_lines, legend_array)
# plt.title(f"success: {100*success/len(data):0.3f}%")
# plt.xlabel("time")
# plt.ylabel("running average performance")


# plt.show()



for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=improvement_adj(init_fit, final_fit)

    #plt.scatter( init_fit, final_fit, color=[improvement,0,0,alpha] )
    plt.scatter( init_fit, final_fit, color=alpha_adj(cmap(improvement) ) )
plt.plot( [0,100], [0,100], color=[.5,.5,.5,0.25] )
plt.title(f"success: {100*success/len(data):0.3f}%")
plt.xlabel("init_fit")
plt.ylabel("final_fit")



plt.legend(custom_lines, legend_array)
plt.xlim(0,.8)
plt.ylim(0,.8)

plt.show()

for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=improvement_adj(init_fit, final_fit)



    #plt.scatter( init_dist, final_dist, color=[final_fit,0,0,0.25] )
    plt.scatter( init_dist, final_dist, color=alpha_adj(cmap(improvement) ) )

x_min = plt.xlim()
y_min = plt.ylim()
plt.xlim(0,x_min[1])
plt.ylim(0,y_min[1])

plt.plot( [0,100], [0,100], color=[.5,.5,.5,0.25] )

plt.title(f"success: {100*success/len(data):0.3f}%")
plt.xlabel("init_dist")
plt.ylabel("final_dist")
plt.legend(custom_lines, legend_array)

plt.show()

for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=improvement_adj(init_fit, final_fit)

    #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
    #plt.scatter( init_dist, final_fit, color=cmap(improvement) )
    plt.arrow( init_dist, init_fit, 0,final_fit-init_fit, color=alpha_adj(cmap(improvement) ), head_width=0.04, head_length=0.01 )

    
    plt.xlabel("init_dist")
    plt.ylabel("final_fit")
    
plt.legend(custom_lines, legend_array)
plt.show()

#ADD PLOT START AND STOP
#######################################
for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    init_w00=data[index][4]
    init_w01=data[index][5]
    init_w10=data[index][6]
    init_w11=data[index][7]

    final_w00=data[index][8]
    final_w01=data[index][9]
    final_w10=data[index][10]
    final_w11=data[index][11]
    improvement=improvement_adj(init_fit, final_fit)

    #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
    #plt.scatter( init_dist, final_fit, color=cmap(improvement) )

    head_width=0.4
    head_length=0.1

    plt.arrow( init_w00, init_w11, final_w00-init_w00, final_w11-init_w11, color=alpha_adj(cmap(improvement) ),\
         head_width=head_width, head_length=head_length )

    
    plt.xlabel("w00")
    plt.ylabel("w11")
    
plt.legend(custom_lines, legend_array)
plt.show()





for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=improvement_adj(init_fit, final_fit)

    #plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
    plt.scatter( init_dist, init_fit, color=alpha_adj(cmap(improvement) ) )
    
    plt.xlabel("init_dist")
    plt.ylabel("init_fit")
    
plt.legend(custom_lines, legend_array)
plt.show()

   
