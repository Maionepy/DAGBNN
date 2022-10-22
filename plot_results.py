import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboard as tb
import pandas as pd
import json

def main():
    env_name = "InvertedDoublePendulum-v2"
    mbpo_available = False
    mnm_available = False
    threshold = 15000
    factor = 10 # 10 for pendulum and 100 for hopper

    sns.set(style="darkgrid")    

    #################### SAC
    color_SAC = "#00FF00"
    with (open(f"elements/sac_elements/{env_name}_SAC_v1.json", "rb")) as f:
        v1 = np.array(json.load(f))
        v1 = v1.tolist()
        while v1[-1][1]*factor < threshold:
            v1.append([v1[-1][0], v1[-1][1] + 10, v1[-1][2]])
        v1 = np.array(v1)
        cond = np.array(v1[:,1])*factor < threshold
        v1 = v1[cond]
    with (open(f"elements/sac_elements/{env_name}_SAC_v2.json", "rb")) as f:
        v2 = np.array(json.load(f))
        v2 = v2.tolist()
        while v2[-1][1]*factor < threshold:
            v2.append([v2[-1][0], v2[-1][1] + 10, v2[-1][2]])
        v2 = np.array(v2)
        cond = np.array(v2[:,1])*factor < threshold
        v2 = v2[cond]
    with (open(f"elements/sac_elements/{env_name}_SAC_v3.json", "rb")) as f:
        v3 = np.array(json.load(f))
        v3 = v3.tolist()
        while v3[-1][1]*factor < threshold:
            v3.append([v3[-1][0], v3[-1][1] + 10, v3[-1][2]])
        v3 = np.array(v3)
        cond = np.array(v3[:,1])*factor < threshold
        v3 = v3[cond]
    with (open(f"elements/sac_elements/{env_name}_SAC_v4.json", "rb")) as f:
        v4 = np.array(json.load(f))
        v4 = v4.tolist()
        while v4[-1][1]*factor < threshold:
            v4.append([v4[-1][0], v4[-1][1] + 10, v4[-1][2]])
        v4 = np.array(v4)
        cond = np.array(v4[:,1])*factor < threshold
        v4 = v4[cond]
    with (open(f"elements/sac_elements/{env_name}_SAC_v5.json", "rb")) as f:
        v5 = np.array(json.load(f))
        v5 = v5.tolist()
        while v5[-1][1]*factor < threshold:
            v5.append([v5[-1][0], v5[-1][1] + 10, v5[-1][2]])
        v5 = np.array(v5)
        cond = np.array(v5[:,1])*factor < threshold
        v5 = v5[cond]

    v = np.array([v1[:,2], v2[:,2], v3[:,2], v4[:,2], v5[:,2]])
    avgV = np.mean(v, axis=0)
    stdSampleV = np.std(v, axis=0, ddof=1)

    cond = np.array(v1[:,1])*factor < threshold

    plt.plot((np.array(v1[:,1])*factor)[cond], avgV[cond], linewidth=1.5, label="SAC", c=color_SAC, marker='o')
    plt.fill_between((np.array(v1[:,1])*factor)[cond], (avgV-stdSampleV)[cond], (avgV+stdSampleV)[cond], color=color_SAC, alpha=0.25)
    

    #################### MBPO
    color_MBPO = '#ff0000'
    if mbpo_available:
        with (open(f"elements/mbpo_elements/{env_name}_mbpo.pkl", "rb")) as f:
            v = pickle.load(f)

        x_axis = np.array(v['x'])*1000
        x_axis = np.insert(x_axis, 0, 0)

        y_axis = np.array(v['y'])
        y_axis = np.insert(y_axis, 0, 0)

        std_axis = np.array(v['std'])
        std_axis = np.insert(std_axis, 0, 0)
        
        cond = x_axis < threshold

        plt.plot((x_axis)[cond], y_axis[cond], linewidth=1.5, label="MBPO", c=color_MBPO, marker='o')
        plt.fill_between((x_axis)[cond], np.array(y_axis-std_axis)[cond], np.array(y_axis+std_axis)[cond], color=color_MBPO, alpha=0.25)
    else:
        with (open(f"elements/{env_name}_reward_MBPO_v1.pkl", "rb")) as f:
            v1 = pickle.load(f)
        with (open(f"elements/{env_name}_reward_MBPO_v2.pkl", "rb")) as f:
            v2 = pickle.load(f)
        with (open(f"elements/{env_name}_reward_MBPO_v3.pkl", "rb")) as f:
            v3 = pickle.load(f)
        with (open(f"elements/{env_name}_reward_MBPO_v4.pkl", "rb")) as f:
            v4 = pickle.load(f)
        with (open(f"elements/{env_name}_reward_MBPO_v5.pkl", "rb")) as f:
            v5 = pickle.load(f)
        
        v = np.array([v1["y"], v2["y"], v3["y"], v4["y"], v5["y"]])

        avgV = np.mean(v, axis=0)
        stdSampleV = np.std(v, axis=0, ddof=1)

        avgV = np.insert(avgV, 0, 0)
        stdSampleV = np.insert(stdSampleV, 0, 0)

        x_axis = np.array(v1['x'])*1000
        x_axis = np.insert(x_axis, 0, 0)

        plt.plot(x_axis, avgV, linewidth=1.5, label="MBPO", c=color_MBPO, marker='o')
        plt.fill_between(x_axis, avgV-stdSampleV, avgV+stdSampleV, color=color_MBPO, alpha=0.25)
        

    #################### DAGBNN
    color_DAGBNN = '#1f77b4'

    with (open(f"elements/{env_name}_reward_DAGBNN_v1.pkl", "rb")) as f:
        v1 = pickle.load(f)
    with (open(f"elements/{env_name}_reward_DAGBNN_v2.pkl", "rb")) as f:
        v2 = pickle.load(f)
    with (open(f"elements/{env_name}_reward_DAGBNN_v3.pkl", "rb")) as f:
        v3 = pickle.load(f)
    with (open(f"elements/{env_name}_reward_DAGBNN_v4.pkl", "rb")) as f:
        v4 = pickle.load(f)
    with (open(f"elements/{env_name}_reward_DAGBNN_v5.pkl", "rb")) as f:
        v5 = pickle.load(f)
    
    v = np.array([v1["y"], v2["y"], v3["y"], v4["y"], v5["y"]])

    avgV = np.mean(v, axis=0)
    stdSampleV = np.std(v, axis=0, ddof=1)

    avgV = np.insert(avgV, 0, 0)
    stdSampleV = np.insert(stdSampleV, 0, 0)

    x_axis = np.array(v1['x'])*1000
    x_axis = np.insert(x_axis, 0, 0)

    plt.plot(x_axis, avgV, linewidth=1.5, label="DAGBNN", c=color_DAGBNN, marker='o')
    plt.fill_between(x_axis, avgV-stdSampleV, avgV+stdSampleV, color=color_DAGBNN, alpha=0.25)
    

    #################### MNN
    color_MNM = "#993399"
    if mnm_available:
        with (open(f"elements/mnm_elements/mujoco_curves.json", "rb")) as f:
            v = json.load(f)
            v = v[env_name]

        plt.plot(np.array(v[0]), np.array(v[1]), linewidth=1.5, label="MnM", c=color_MNM, marker='o')
        plt.fill_between(np.array(v[0]), np.array(v[1])-np.array(v[2]), np.array(v[1])+np.array(v[2]), color=color_MNM, alpha=0.25)
        

    #################### END
    plt.legend(loc='upper left')
    #plt.grid()
    xstick = np.linspace(0, threshold, int(threshold/1000) + 1 , endpoint=True)
    xstick = xstick.tolist()
    new_xstick = []
    list_remove = []
    for elem in xstick:
        if int(elem/1000) == 0:
            new_xstick.append('0')
        elif int(elem/1000) in [5,10,15,25,50,75,100,125,150,200,300]:
            new_xstick.append(str(int(elem/1000)) + "k")
        else:
            list_remove.append(elem)
    for remove_item in list_remove:
        xstick.remove(remove_item)
    plt.xticks(xstick, new_xstick)  # Set text labels and properties.


    # Add title and axis names
    # plt.title(env_name[:-3],fontsize=18)
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Average return', fontsize=18)
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=threshold)

    plt.show()
    return

if __name__ == '__main__':
    main()