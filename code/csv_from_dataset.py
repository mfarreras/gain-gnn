from datanetAPI import DatanetAPI
import networkx as nx
import csv
import numpy as np

def generate_csvs(path, filename):
    with open('csv_analysis/paths_'+filename, 'w') as paths:
        with open('csv_analysis/links_'+filename, 'w') as links:
            write_paths = csv.writer(paths)
            write_links = csv.writer(links)
            header_paths = ['PktsDrop','AvgDelay','AvgLnDelay','AvgBw','PktsGen','pathLength','delay_real','delay_calculat']
            header_links = ['bandwidth','queue_sizes','utilization','losses','avgPacketSize','avgPortOccupancy',
                            'maxQueueOccupancy','occupancy','offeredTrafficIntensity']
            #ADD HEADER TO FILES
            tool = DatanetAPI(path, shuffle=False)
            it = iter(tool)
            for s in it:
                G = nx.DiGraph(s.get_topology_object().copy())
                R = s.get_routing_matrix()
                T = s.get_traffic_matrix() 
                D = s.get_performance_matrix()
                P = s.get_port_stats()

                #AvgBw_sum calculation
                AvgBw_sum = np.zeros((len(G.nodes),len(G.nodes)))
                for src in range(G.number_of_nodes()):
                    for dst in range(G.number_of_nodes()):
                        route = R[src,dst]
                        if route:
                            for index, node in enumerate(route):
                                    next_node_index = index + 1
                                    if next_node_index < len(route):
                                        for f_id in range(len(T[src, dst]['Flows'])):
                                            if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                                                AvgBw_sum[node][route[next_node_index]] += T[src, dst]['Flows'][f_id]['AvgBw']

                #occupancy list src, dst 
                occupancy = np.zeros((len(G.nodes),len(G.nodes)))
                for edge in G.edges:
                    src, dst = edge
                    occupancy[src][dst] = P[src][dst]['qosQueuesStats'][0]['avgPortOccupancy']/G.nodes[src]['queueSizes']
                
                #REPASSAR CALCUL DELAY
                for src in range(G.number_of_nodes()):
                    for dst in range(G.number_of_nodes()):
                        if src != dst:
                            for f_id in range(len(T[src, dst]['Flows'])):
                                if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                                    #obtenir ruta src-dst i calcular delay!!!
                                    route = R[src,dst]
                                    delay_path = 0
                                    for index, node in enumerate(route):
                                        next_node_index = index + 1
                                        if next_node_index < len(route):
                                            if occupancy[node][route[next_node_index]] < 0:
                                                occupancy[node][route[next_node_index]] = 0
                                            elif occupancy[node][route[next_node_index]] > 1:
                                                occupancy[node][route[next_node_index]] = 1

                                            delay = occupancy[node][route[next_node_index]]*G.nodes[node]['queueSizes']*T[src, dst]['Flows'][f_id]['SizeDistParams']['AvgPktSize']/s.get_srcdst_link_bandwidth(node,route[next_node_index]) #Cal agafar avgpktsize de tots els flows

                                            #sum results per path
                                            delay_path += delay
                            path_sample = [
                                D[src,dst]["AggInfo"]["PktsDrop"],
                                D[src,dst]["AggInfo"]["AvgDelay"],
                                D[src,dst]["AggInfo"]["AvgLnDelay"],
                                T[src,dst]["AggInfo"]["AvgBw"],
                                T[src,dst]["AggInfo"]["PktsGen"],
                                len(R[src,dst]),
                                D[src, dst]['Flows'][0]['AvgDelay'],
                                delay_path #crear percentatge de resta en valor absolut, dividir-ho pel correcte (delay real-delay processat)/delay real en valor absolut
                            ]
                            #print(path_sample)
                            if D[src, dst]['Flows'][0]['AvgDelay'] >= 0:
                                write_paths.writerow(path_sample)

                            try:
                                #Comprovar si hi ha link primer!!!
                                bandwidth = s.get_srcdst_link_bandwidth(src,dst)
                                queueSizes = G.nodes[src]['queueSizes']
                                link_sample = [         
                                    bandwidth,
                                    queueSizes,
                                    P[src][dst]["qosQueuesStats"][0]["utilization"],
                                    P[src][dst]["qosQueuesStats"][0]["losses"],
                                    P[src][dst]["avgPacketSize"],
                                    P[src][dst]["qosQueuesStats"][0]["avgPortOccupancy"],
                                    P[src][dst]["qosQueuesStats"][0]["maxQueueOccupancy"],
                                    occupancy[src][dst],
                                    AvgBw_sum[src][dst]/bandwidth#offeredTrafficIntensity
                                ]
                                #print(link_sample)
                                write_links.writerow(link_sample)
                            except KeyError:
                                continue
                            except IndexError:
                                continue

train = ['']#, '25', '50']

validation = ['']#, '50', '300']
#validation = ['50', '300']

"""for dataset in train:
    generate_csvs('../data/gnnet-ch21-dataset-train/'+dataset, 'train_calculated'+dataset+'.csv')"""

for dataset in validation:
    generate_csvs('../data/gnnet-ch21-dataset-validation', 'validation_calculated.csv')

"""for dataset in validation:
generate_csvs('../data/gnnet-ch21-dataset-test/ch21-test-setting-2', 'validationS2_calculated.csv')

for dataset in validation:
generate_csvs('../data/gnnet-ch21-dataset-test/ch21-test-setting-3', 'validationS3_calculated.csv')"""