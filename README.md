# Robust Max Network Flows
This project studies the problem of finding [maximum network flows](https://en.wikipedia.org/wiki/Maximum_flow_problem) from the perspective of robustness and adaptivity. Understanding robust versions of 
classical network problems is an important task because most real-world networks−-especially those for which we might desire to compute network flows−-
are susceptible to failure and uncertainty. We implement algorithms for the formulation of arc-based robust and adaptive flow done in [[1]](#1), along with a 
shortest augmenting path algorithm [[2]](#2) to solve the classical max flow problem. We also create approximation algorithms for the robust and adaptive max flows. Detailed 
documentation and examples are included in [code.jl](https://github.com/dakshces/RobustMaxNetFlows/blob/main/code.jl). Algorithms can be tested on flow networks in [dat.txt](https://github.com/dakshces/RobustMaxNetFlows/blob/main/dat.txt) generated using NETGEN [[3]](#3); parameters used to generate networks are in [par.txt](https://github.com/dakshces/RobustMaxNetFlows/blob/main/par.txt). Methods to generate NETGEN networks are included.


## References

<a id="1">[1]</a> 
Dimitris Bertsimas, Ebrahim Nasrabadi, and Sebastian Stiller. “Robust and Adaptive Network Flows”. In: Operations Research 61.5 (2013), pp. 1218–1242. issn: 0030- 364X.

<a id="2">[2]</a> 
Ravindra K. Ahuja, Thomas L. Magnanti, and James B. Orlin. Network Flows: Theory, Algorithms, and Applications. New Jersey: Prentice Hall, 1993.

<a id="3">[3]</a> 
D. Klingman, A. Napier, and J. Stutz. “NETGEN: A Program for Generating Large Scale Capacitated Assignment, Transportation, and Minimum Cost Flow Network Problems”. In: Management Science 20.5 (1974), pp. 814–821. issn: 0025-1909.
