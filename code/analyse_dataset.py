"""
   Copyright 2021 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from datanetAPI import DatanetAPI
import pandas as pd
import networkx as nx
import json

tool = DatanetAPI("../data/gnnet-ch21-dataset-validation/setting-1-300", shuffle=False)
it = iter(tool)
num_samples = 0

dsample = []
for s in it:
    G = s.get_topology_object().copy()

    R = s.get_routing_matrix()
    longest_routing = 0
    for elem in R:
        for e in elem:
            if e and longest_routing < len(e):
                longest_routing = len(e)

    T = s.get_traffic_matrix() #Returns the traffic_matrix. Assuming this matrix is denoted by m, the information that traffic_matrix stores for a specific src-dst pair can be accessed using m[src,dst] . See more details about the traffic_matrix in the previous section.
    D = s.get_performance_matrix() #Returns the performance_matrix. Assuming this matrix is denoted by m, performance measurements of a specific src-dst pair can be accessed using m[src,dst]. See more details about the performance_matrix in the previous section.
    P = s.get_port_stats() #Returns the port_stats object. Assuming this object is denoted by ps, the information that port_stats stores for a specific src-dst port can be accessed using ps[src][dst] . See more details about the port_stats in the previous section.

    for src in range(len(T)):
        for dst in range(len(T)):
            #Si link a P[src][dst], desar P, altrament none
            if dst in P[src]:
                avgPacketSize = P[src][dst]["avgPacketSize"]
                utilization = P[src][dst]["qosQueuesStats"][0]["utilization"]
                losses = P[src][dst]["qosQueuesStats"][0]["losses"]
                avgPacketSize = P[src][dst]["qosQueuesStats"][0]["avgPacketSize"]
                avgPortOccupancy = P[src][dst]["qosQueuesStats"][0]["avgPortOccupancy"]
                maxQueueOccupancy = P[src][dst]["qosQueuesStats"][0]["maxQueueOccupancy"]
            else:
                avgPacketSize = None
                utilization = None
                losses = None
                avgPacketSize = None
                avgPortOccupancy = None
                maxQueueOccupancy = None

            sample = { 'network_size' : s.get_network_size(), #Returns the number of nodes in the topology.
                'number_of_links': len(G.edges), #ENTRAR numero NODOS Y LINKS
                'diameter': nx.diameter(G), #Calcular diàmetre i paràmetre K per normalitzar (mirar màxim-mínim?)
                'longest_routing' : longest_routing,
                'num_packets': s.get_global_packets(), #Returns the number of packets transmitted in the network per time unit of the sample.
                'global_losses': s.get_global_losses(), #Returns the number of packets dropped in the network per time unit of the sample.
                'global_delay': s.get_global_delay(), #Returns the average per-packet delay over all the packets transmitted in the network in time units of the sample.
                'max_avg_lambda': s.get_maxAvgLambda(), #Returns the maximum average lambda selected to generate the traffic matrix of the sample.
                #D (performance_matrix)
                'src': src,
                'dst': dst,
                #'AggInfo': 
                'PktsDrop': D[src,dst]["AggInfo"]["PktsDrop"],
                'AvgDelay': D[src,dst]["AggInfo"]["AvgDelay"],
                'AvgLnDelay': D[src,dst]["AggInfo"]["AvgLnDelay"],
                'p10': D[src,dst]["AggInfo"]["p10"],
                'p20': D[src,dst]["AggInfo"]["p20"],
                'p50': D[src,dst]["AggInfo"]["p50"],
                'p80': D[src,dst]["AggInfo"]["p80"],
                'p90': D[src,dst]["AggInfo"]["p90"],
                'Jitter': D[src,dst]["AggInfo"]["Jitter"],

                #Recòrrer T (traffic_matrix)
                #'AggInfo':
                'AvgBw': T[src,dst]["AggInfo"]["AvgBw"],
                'PktsGen': T[src,dst]["AggInfo"]["PktsGen"],

                #Recòrrer P (port_stats)
                'avgPacketSize': avgPacketSize,
                #'qosQueuesStats': 
                'utilization': utilization,
                'losses': losses,
                'avgPacketSize': avgPacketSize,
                'avgPortOccupancy': avgPortOccupancy,
                'maxQueueOccupancy': maxQueueOccupancy
            }
            #Si flow(s) a T, desar flows, altrament none
            if len(T[src,dst]["Flows"]) > 0:
                for flowT,flowD in zip(T[src,dst]["Flows"],D[src,dst]["Flows"]):
                    subsample = {
                        #'Flows'[]: 
                        'PktsDrop': flowD["PktsDrop"],
                        'AvgDelay': flowD["AvgDelay"],
                        'AvgLnDelay': flowD["AvgLnDelay"],
                        'p10': flowD["p10"],
                        'p20': flowD["p20"],
                        'p50': flowD["p50"],
                        'p80': flowD["p80"],
                        'p90': flowD["p90"],
                        'Jitter': flowD["Jitter"],

                        #'Flows'[]:
                        'TimeDist':flowT["TimeDist"],
                        #'TimeDistParams':
                            'EqLambda': flowT["TimeDistParams"]["EqLambda"],
                            'AvgPktsLambda':flowT["TimeDistParams"]["AvgPktsLambda"],
                            'ExpMaxFactor':flowT["TimeDistParams"]["ExpMaxFactor"],
                        'SizeDist':flowT["SizeDist"],
                        #'SizeDistParams':
                            'AvgPktSize':flowT["SizeDistParams"]["AvgPktSize"],
                            'PktSize1':flowT["SizeDistParams"]["PktSize1"],
                            'PktSize2':flowT["SizeDistParams"]["PktSize2"],
                        'AvgBw':flowT["AvgBw"],
                        'PktsGen':flowT["PktsGen"],
                        'ToS':flowT["ToS"]
                    }
                    dsample.append({**sample, **subsample})
            else:
                dsample.append(sample)
    num_samples += 1
df_samples = pd.DataFrame(dsample)

tfile = open('test.txt', 'a')
tfile.write(df_samples.to_string())
tfile.close()


