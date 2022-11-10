import os
import tensorflow as tf
import numpy as np
from glob import iglob
import configparser
from itertools import zip_longest
import zipfile
from read_dataset import input_fn
from datanetAPI import DatanetAPI
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FILENAME = 'upload_file_offered_traffic_pktsgen' #CANVIARRRRRRRRRRRRRRRRRRRRRRRRR###########################################################################

# Remember to change this path if you want to make predictions on the final
# test dataset -> './utils/paths_per_sample_test_dataset.txt'
PATHS_PER_SAMPLE = './utils/paths_per_sample_test_dataset.txt'

upload_file = open(FILENAME+'.txt', "w")

# Read the config.ini file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config_submission.ini') 

directories = [d for d in iglob(config['DIRECTORIES']['test'] + '/*/*')]
# First, sort by scenario and second, by topology size
directories.sort(key=lambda f: (os.path.dirname(f), int(os.path.basename(f))))
first = True
results = []
with open('outfile', 'rb') as fp:
    pred = pickle.load(fp)
    for result in pred:
        for d in directories:
            print('Current directory: ' + d)

            # It is NECESSARY to keep shuffle as 'False', as samples have to be read always in the same order
            ds_test = input_fn(d, shuffle=False)
            ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
            
            tool = DatanetAPI(d, shuffle=False)
            it = iter(tool)
            for sample in it:
                G = sample.get_topology_object().copy()
                T = sample.get_traffic_matrix()
                R = sample.get_routing_matrix()

                #create src, dst occupancy list
                occupancy = np.zeros((len(G.nodes),len(G.nodes)))
                i = 0
                for edge in G.edges:
                    src, dst, t = edge
                    occupancy[src][dst] = result[i]
                    i+=1

                for src in range(G.number_of_nodes()):
                    for dst in range(G.number_of_nodes()):
                        if src != dst:
                            for f_id in range(len(T[src, dst]['Flows'])):
                                if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                                    #obtenir ruta src-dst i calcular delay!!!
                                    route = R[src,dst]
                                    delay_route = 0
                                    for index, node in enumerate(route):
                                        next_node_index = index + 1
                                        if next_node_index < len(route):
                                            #for each source - destination find occupation in results (sorted as it was not shuffled)
                                            #compute delay

                                            if occupancy[node][route[next_node_index]] < 0:
                                                occupancy[node][route[next_node_index]] = 0
                                            elif occupancy[node][route[next_node_index]] > 1:
                                                occupancy[node][route[next_node_index]] = 1

                                            delay = occupancy[node][route[next_node_index]]*(G.nodes[node]['queueSizes']*T[src, dst]['Flows'][f_id]['SizeDistParams']['AvgPktSize'])/G[node][route[next_node_index]][0]['bandwidth'] #Cal agafar avgpktsize de tots els flows

                                            #sum results per path
                                            delay_route += delay
                                    # save to results
                                    results.append(delay_route)
            # Separate predictions of each sample; each line contains all the per-path predictions of that sample
            # excluding those paths with no traffic (i.e., flow['AvgBw'] != 0 and flow['PktsGen'] != 0)
            idx = 0
            for x, y in ds_test:
                top_pred = results[idx: idx+int(x['n_paths'])]
                idx += int(x['n_paths'])
                if not first:
                    upload_file.write("\n")
                upload_file.write("{}".format(';'.join([format(i,'.6f') for i in np.squeeze(top_pred)])))
                first = False

upload_file.close()

zipfile.ZipFile(FILENAME+'.zip', mode='w').write(FILENAME+'.txt')

########################################################
###### CHECKING THE FORMAT OF THE SUBMISSION FILE ######
########################################################
sample_num = 0
error = False
print("Checking the file...")

with open(FILENAME + '.txt', "r") as uploaded_file, open(PATHS_PER_SAMPLE, "r") as path_per_sample:
    # Load all files line by line (not at once)
    for prediction, n_paths in zip_longest(uploaded_file, path_per_sample):
        # Case 1: Line Count does not match.
        if n_paths is None or prediction is None:
            print("WARNING: File must contain 1560 lines in total for the final test datset (90 for the toy dataset). "
                  "Looks like the uploaded file has {} lines".format(sample_num))
            error = True

        # Remove the \n at the end of lines
        prediction = prediction.rstrip()
        n_paths = n_paths.rstrip()

        # Split the line, convert to float and then, to list
        prediction = list(map(float, prediction.split(";")))

        # Case 2: Wrong number of predictions in a sample
        if len(prediction) != int(n_paths):
            print("WARNING in line {}: This sample should have {} path delay predictions, "
                  "but it has {} predictions".format(sample_num, n_paths, len(prediction)))
            error = True

        sample_num += 1

if not error:
    print("Congratulations! The submission file has passed all the tests! "
          "You can now submit it to the evaluation platform")