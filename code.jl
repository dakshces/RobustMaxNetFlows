using LightGraphs, SimpleWeightedGraphs
using DataStructures, SparseArrays, LinearAlgebra
using JuMP, GLPK
using StatsBase, Distributions, Random
using GraphPlot, Cairo, Compose
using Plots

#=
Author: Daksh Aggarwal
Code for project "Robust Max Flows"
Date: 18 October 2020

Note: Some simple examples to call different functions and running experiments are 
provided below. Generating data shown in report takes a while (about an hour), so 
we have given examples with small values of parameters.
=#
#------------------------------EXAMPLES------------------------------------#
#=
include("code.jl")

****Try different max flows****

G = SimpleWeightedDiGraph(
    [0 16 13 0 0 0; 
     0 0 6 12 0 0; 
     0 0 0 0 14 0; 
     0 0 9 0 0 20; 
     0 0 0 7 0 4; 
     0 0 0 0 0 0]);


f1,x1 = shortestpath_augment_maxflow(G,1,6);
plot_flow(G,x1,"classical.pdf") #saves plot showing network flow to classical.pdf

f2,x2 = robust_maxflow(G,[0,1,0,0,0,0],1,6);
plot_flow(G,x2, "robust.pdf") 

f3,x3 = adaptive_maxflow_sampling(G, 1, 1, 6, 1);
plot_flow(G,x3, "adaptive.pdf")  

f4,x4 = newton_maxflow(G,1,6);
plot_flow(G,x4, "newton.pdf")

****Try experiments****
run_exper1("dat.txt", 100, 140, 20, 10, 1, 50, "optimality_ratio")

run_exper2("dat.txt",100, 140, 20, 10, 50, 0.1, "vanishing_gamma")

run_exper3("dat.txt",100, 140, 20, 10, 1, 50, 0.1, "capacity_violation")

run_exper4("dat.txt", 100, 140, 20, 10, 1, 50,"similarities")

***Generating new NETGEN networks***

generate_netgenNetworks("par2.txt", "dat2.txt", 100, 1000, 20, 10, 1000, 10000)

# generates NETGEN paramters and saves to par2.txt, to generate 10 networks with sizes 100, 120, 140, ..., 1000, and each edge having min capacity 1000 and max capacity 10000. Networks saved to dat2.txt
=#


#------------------------------NETWORK GENERATION--------------------------#

#--------PARAMETER GENERATION & CALLING NETGEN---------#
#=
Procedure: generate_netgenNetworks
Purpose: Generate network paramters (random number of edges) and generate networks by passing parameters to NETGEN
Parameters: par_file, a string, name of file to save generated paramters for NETGEN
            data_file, a string, name of file to save generated networks  
            minNodes, int, min number of nodes in a network
            maxNodes, int, max number of nodes in a network 
            nodeStep, int, steps at which to generate networks in [minNodes,maxNodes]
            each, int, number of networks to generate for each network size
            minCap, float, minimum capacity of network
            maxCap, float, maximum capacity of network
Produces: none
Preconditions: none
Postconditions: Writes parameters to par_file and networks to data_file
=#
function generate_netgenNetworks(par_file, data_file, minNodes, maxNodes, nodeStep, each, minCap, maxCap)
    generate_parameters(par_file, minNodes, maxNodes, nodeStep, each, minCap, maxCap)
    parameters = read(par_file, String)
    call_netgen(par_file, data_file)
end

#=
Procedure: generate_parameters
Purpose: Generate network paramters for specified range of network size and edge capacities; generated parameters are in NETGEN format and will generate max flow problems
Parameters: par_file, a string, name of file to save generated paramters for NETGEN
            minNodes, int, min number of nodes in a network
            maxNodes, int, max number of nodes in a network 
            nodeStep, int, steps at which to generate networks in [minNodes,maxNodes]
            each, int, number of networks to generate for each network size
            maxEdgeFrac, float, max fraction of all possible of edges (n(n-1)/2) to randomly sample number of edges; controls density of networks 
            minCap, float, minimum capacity of network
            maxCap, float, maximum capacity of network
Produces: none
Preconditions: Uses PRNG seed 13502460 (NETGEN standard)
Postconditions: Writes parameters to par_file
=#
function generate_parameters(par_file, minNodes, maxNodes, nodeStep, each, minCap, maxCap)
    prob = 0
    open(par_file, "w") do f
        for n in minNodes:nodeStep:maxNodes
            for i in 1:each
                prob += 1
                edges = max(2*n, floor(Int64,rand(Float64)*10*n))
                write(f, "13502460\n$prob $n 1 1 $edges 1 1 1 0 0 0 100 $minCap $maxCap\n")
            end
        end
    end
end

#=
Procedure:call_netgen
Purpose: Pass parameters to NETGEN executable and save generated networks
Parameters: par_file, a string, name of file with paramters for NETGEN
            data_file, a string, name of file to save generated networks
Produces: none
Preconditions: NETGEN executable is named "./netgen_exe"
Postconditions: Writes networks to data_file
=#
function call_netgen(par_file, data_file)
    run(pipeline(stdin=par_file,`./netgen_exe`, stdout=data_file))
end


#-----------------READING NETWORKS FROM FILE------------------#

#=
Struct: netgenNetworks
Purpose: facilitate creation of an iterator over generated NETGEN networks to conseve memory
Fields: total, int, number of networks to read from start of file
        f, IOStream, open stream to file containing NETGEN networks
=#
struct netgenNetworks
    total::Int
    f::IOStream
end


#=
Purpose: Defines iterator for netgenNetworks struct; reads NETGEN networks from file
Behaviour: For a netgenNetworks instance data, generates data.total number of networks using stream data.f; closes data.f once all S.total networks are read
Produces: iteratively generates SimpleWeightedDiGraphs
Preconditions: S.f is valid open IO stream to file with networks generated using NETGEN
Postconditions: none
=#
Base.iterate(data::netgenNetworks, state=1) = state > data.total ?
                                              close(data.f) :
                                              (get_next(data.f), state+1)


#=
Procedure: get_next
Purpose: read next NETGEN network after current position of data file IOStream
Parameters: f, IOStream, open stream to file containing NETGEN networks
Produces: SimpleWeightedDiGraph
Preconditions: networks in file are in NETGEN format
Postconditions: combines any anti-parallel edges between nodes (no loss of generality)
=#    
function get_next(f::IOStream)
    n = 0
    
    for l in eachline(f)
        if occursin("Number of nodes:", l)
            n = parse(Int64,split(l)[end])
            break
        end
    end
    
    for l in eachline(f)
        if occursin(r"t$", l)
            break
        end
    end
    
    srcs = Int64[]
    dsts = Int64[]
    caps = Float64[]
    
    for l in eachline(f)
        if occursin("NETGEN", l)
            break
        end
        sdc = map(x->parse(Int64,x),split(l)[2:4])
        push!(srcs,sdc[1])
        push!(dsts,sdc[2])
        push!(caps,sdc[3])
    end
    
    infCap = n*maximum(caps) ## infinite capacity for purpose of current network
    
    push!(srcs, n+1) 
    push!(dsts, 1)
    push!(caps, infCap) ## add super source (to avoid cycles)
    
    push!(srcs, n)
    push!(dsts, n+2)
    push!(caps, infCap) ## add super sink (to avoid cycles)
    
    w = sparse_adjacency(n+2, srcs, dsts, caps)
    return SimpleWeightedDiGraph(w)
end


#=
Procedure: sparse_adjacency
Purpose: Create sparse adjacency matrix representation of network
Parameters: n, int, number of nodes in network
            srcs, ordered list of sources
            dsts, ordered list of destinations
            caps, ordered list of capacities
Produces: n x n SparseMatrixCSC
Preconditions: for each i, (srcs[i],dsts[i],caps[i]) represents edge from srcs[i] to dsts[i] with capacity caps[i]
Postconditions: combines any anti-parallel edges between nodes (no loss of generality)
=#
function sparse_adjacency(n, srcs, dsts, caps)
    w = spzeros(n,n)
    for (i,j,c) in zip(srcs, dsts, caps)
        w[i,j] = c
    end
     for i in 1:n
        for j in (i+1):n
            if w[j, i] != 0
                if w[i,j] > w[j,i]
                    w[i,j] -= w[j,i]
                    w[j,i] = 0 
                else
                    w[j,i] -= w[i,j]
                   w[i,j] = 0
                end
            end
        end
     end
    dropzeros!(w)
    return w
end


#-----------------------MAX FLOW (CLASSICAL SOLUTION)----------------------#

#-----------SHORTEST AUGMENTING PATH ALGORITHM----------#
#=
Procedure: shortestpath_augment_maxflow
Purpose: Uses shortest augmentating paths to find
         a maximum flow in the network G from s to t
Parameters: G, a SimpleWeightedDiGraph
            s, the source vertex
            t, the sink vertex
Produces: f, the maximum flow value from s to t,
          x, the flow in the network (n x n matrix)
          r, the residual network (as n x n matrix)
Preconditions: * weight of an edge in G denotes it capacity
Postconditions: none 
=#
function shortestpath_augment_maxflow(G,s,t)
    n = nv(G)
    m = ne(G)
    r = copy(LightGraphs.weights(G)) #residual network
    d = distance_labels(G,t) 
    pred = Dict{Int,Int}()
    i = s
    prev = -1
    maxiter = m*n
    iter = 0
    while d[s] < n && iter < maxiter
        j = 0
        iter += 1
        for k in outneighbors(G,i)
            if r[i,k] > 0 && d[i] == d[k]+1 && (i == s || prev != k)
                j = k
                break
            end
        end
        if j == 0
            if i == s && sum(r[1,:]) == 0 #checks for saturation of source edges
                break
            end
            retreat(G,r,d,i)
            if i != s
                prev = i
                i = pred[i]
            end
        else
            pred[j] = i
            i = j
            if i == t
                augment(G,r,d,pred,s,t)
                i = s
            end
        end
    end
    
    x = zeros((n,n))
    for i in 1:n
        for j in outneighbors(G,i)
            x[i,j] = r[j,i]
        end
    end
    return sum(x[:,t]), sparse(x), r
end


#=
Procedure: distance_labels
Purpose: Constructs backward shortest distance labels for vertices
         using sink vertex t with a reverse BFS
Parameters: G, a SimpleWeightedDiGraph
            t, the sink vertex
Produces: d, vector of shortest-distance labels
Preconditions: t is a valid vertex of G
Postconditions: none 
=#
function distance_labels(G,t)
    T = bfs_tree(G,t,dir=:in) #backward BFS
    n = nv(G)
    d = zeros(Int,n)
    d[t] = 0
    q = Queue{Int}()
    enqueue!(q,t)
    while length(q)!=0
        i = dequeue!(q)
        for j in outneighbors(T,i)
            d[j] = d[i] + 1
            enqueue!(q,j)
        end
    end
    return d
end

#=
Procedure: retreat
Purpose: To update distance label of a vertex 
         when it has no admissible out-edges  
Parameters: G, a SimpleWeightedDiGraph
            r, residual network of G
            d, distance labels
            i, a vertex of G
Produces: none
Preconditions: none
Postconditions: Updates the distance label of vertex i
=#
function retreat(G,r,d,i)
    accessible = filter(k->r[i,k]>0, outneighbors(G,i))
    if accessible == []
        accessible = filter(k->r[i,k]>0, inneighbors(G,i))
    end
    
    if accessible == []
        d[i] = nv(G)+1
    else
        d[i] = 1 + minimum(map(k->d[k], accessible))
    end
        
end

#=
Procedure: augment
Purpose: To find the minimal residue edge along the current path s to t
         and increment the path flows with the min value
Parameters: G, a SimpleWeightedDiGraph
            r, residual network of G
            d, distance labels
            pred, dictionary of most recent predecessors of vertices
            s, the source vertex
            t, the sink vertex
Produces: none
Preconditions: none
Postconditions: Updates the s-t path represented by pred
                in the residual network r
=#
function augment(G,r,d,pred,s,t)
    len = d[s]+1
    path = zeros(Int,len)
    path[len] = t
    for pos in len:-1:2
        path[pos-1] = pred[path[pos]]
    end
    min_res = minimum(map(i->r[path[i],path[i+1]],1:(len-1)))
    for pos in 1:(len-1)
        r[path[pos],path[pos+1]] -= min_res
        r[path[pos+1],path[pos]] += min_res
    end
end


#-----------MINIMUM CUT FROM MAX FLOW----------#

#=
Procedure: min_cut
Purpose: Find edges of the minimum cut from max flow specified by r
Parameters: G, a SimpleWeightedDiGraph
            r, residual network of G after solving max problem
            s, the source vertex
Produces: cut, list of edges (as pairs of nodes) in min cut
Preconditions: r is derived from solving a classical max flow problem
Postconditions: none
=#
function min_cut(G,r,s)
    n = nv(G)
    q = Queue{Int}()
    enqueue!(q,s)
    visited = zeros(Int,n)
    while length(q)!=0
        i = dequeue!(q)
        for j in 1:n
            if r[i,j] != 0  && visited[j] == 0
                visited[j] = 1
                enqueue!(q,j)
            end
        end
    end

    cut = []
    for i in 1:n
        for j in outneighbors(G,i)
            if visited[i] == 1 && visited[j] == 0
                push!(cut,(i,j))
            end
        end
    end
    return cut
end


#----------------------------ROBUST MAX FLOW-----------------------------#

#-----ROBUST FLOW WITH NODE-WISE UNCERTAINTY PARAMETERS------#

#=
Procedure: robust_maxflow
Purpose: Finds maximum robust flow as defined in
         "Robust and Adaptive Network Flows" by Bertsimas et al.
Parameters: G, a SimpleWeightedDiGraph
            gam_v, vector of max # (can be 0) of in-edges that can fail at node v
            s, the source vertex
            t, the sink vertex
Produces: f, the maximum robust flow value from s to t,
          x, the flow in the network (n x n matrix)
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function robust_maxflow(G,gam_v,s,t)
    n = nv(G)
    m = ne(G)
    c = LightGraphs.weights(G) #capacity matrix
    edgl = map(e->(e.src,e.dst),edges(G)) #edge-list
    
    rf = Model(GLPK.Optimizer)
    
    @variable(rf, x[edgl] >= 0) #flow 
    @variable(rf, th[edgl] >= 0) #theta
    @variable(rf, d[setdiff(vertices(G),s)] >= 0) #delta
    
    @constraint(rf, [e=edgl], x[e] <= c[e[1],e[2]]) #flow cannot exceed capacity
    @constraint(rf, [e=edgl], x[e] - th[e] - d[e[2]] <= 0)
    @constraint(rf, [i=setdiff(vertices(G),s)],
                sum([x[e]-th[e] for e ∈ inedges(G,i)]) -
                sum([x[e] for e ∈ outedges(G,i)]) -
                gam_v[i]*d[i] >= 0)   #robust weak flow conservation
    
    @objective(rf, Max,sum([x[e] for e ∈ inedges(G,t)])) #sum of in-flow into sink t
    optimize!(rf)
    
    optx = value.(x)
    xflow = spzeros(n,n)
    for e in edgl
        xflow[e[1],e[2]] = optx[e]
    end
    return objective_value(rf), xflow
end

#=
Procedure: inedges
Purpose: Finds all inedges of j in G
Parameters: G, a SimpleWeightedDiGraph
            j, a vertex
Produces: a list of vertex pairs representing inedges of j
Preconditions: none
Postconditions: none 
=#
function inedges(G,j)
     return map(i->(i,j), inneighbors(G,j))
end

#=
Procedure: outedges
Purpose: Finds all outedges of i in G
Parameters: G, a SimpleWeightedDiGraph
            i, a vertex
Produces: a list of vertex pairs representing outedges of i
Preconditions: none
Postconditions: none 
=#
function outedges(G,i)
    return map(j->(i,j), outneighbors(G,i))
end


#----ROBUST FLOW WITH GLOBAL UNCERTAINTY PARAMETER (APPROX.)-------#

#=
Procedure: robust_maxflow_globalapprox
Purpose: Finds approximation to *global* version of maximum robust flow, which is defined in "Robust and Adaptive Network Flows" by Bertsimas et al.; samples on all weak compositions of gam to obtain approximation and solves corresponding robust flow problem
Parameters: G, a SimpleWeightedDiGraph
            gam, max # of edges which globally fail in network at a time
            s, the source vertex
            t, the sink vertex
            Nsamples (optional), number of samples 
Produces: f, approximate robust max flow value
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function robust_maxflow_globalapprox(G,gam,s,t,Nsamples=-1)
    n = nv(G)
    gam_vs = weak_compositions(gam, n, 10*Nsamples)
    l = length(gam_vs)
    Nsamples = Nsamples < 0 ? l : min(l, Nsamples)
    choices = sample(1:l, Nsamples, replace = false)
    rflow = Inf
    rx = []
    for i in choices
        f,x = robust_maxflow(G, gam_vs[i], s, t)
        if f < rflow
            rflow  = f
            rx = x
        end
    end
   return rflow, rx
end

#=
Procedure: weak_compositions
Purpose: Find N weak compositions of integer n into t parts
    https://en.wikipedia.org/wiki/Composition_(combinatorics)
Parameters: n, int, integer to be decomposed
            t, int, number of parts
            N, int, number of weak compositions to find
Produces: a list of at most N weak compositions
Preconditions: all parameters are positive
Postconditions: none 
=#
function weak_compositions(n, t, N)
    comps = compositions(n+t, t, N)
    for comp in comps
        map(i->comp[i]-=1, 1:t)
    end
    return comps
end

#=
Procedure: compositions
Purpose: Find N compositions of integer n into t parts
    https://en.wikipedia.org/wiki/Composition_(combinatorics)
Parameters: n, int, integer to be decomposed
            t, int, number of parts
            N, int, number of compositions to find
Produces: a list of at most N compositions
Preconditions: all parameters are positive
Postconditions: none 
=#
function compositions(n, t, N)
    combs = combinations(n-1, t-1, N)
    comps = []
    for comb in combs
        lst = zeros(Int, t)
        lst[1] = comb[1]+1
        for i in 2:(t-1)
            lst[i] = comb[i] - comb[i-1]
        end
        lst[t] = n - comb[t-1] - 1
        push!(comps, lst)
    end
    return comps
end

#=
Procedure: combinations
Purpose: Find N k-combinations of {0,1,...,n-1}
Parameters: n, int, number of symbols
            t, int, size of subset
            N, int, number of combinations to find
Produces: a list of at most N combinations
Preconditions: all parameters are positive
Postconditions: none 
=#
function combinations(n, t, N)
    combs = []
    lst = [i for i in 0:(t-1)]
    push!(lst, n)
    push!(lst, 0)
    count = 0
    while count < N
        push!(combs, lst[1:t])
        count += 1
        j = 0
        while lst[j+1] + 1 == lst[j + 2]
            lst[j+1] = j
            j += 1
        end
        
        if(j + 1 > t)
            break
        end
        
        lst[j+1] += 1
    end
    return combs
end


#--------------ROBUST FLOW WITH GLOBAL Γ=1-----------------#

#=
Procedure: newton_maxflow
Purpose: Finds maximum robust flow when at most 1 edge can fail
         "Maximizing Residual Flow under an Arc Destruction" by Aneja et al.
         Uses Newton's Optimation method and Max-Flow Min-Cut Thm.
Parameters: G, a SimpleWeightedDiGraph
            s, the source vertex
            t, the sink vertex
Produces: fλ, the maximum robust flow value from s to t,
          xλ, the flow in the network (n x n matrix)
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function newton_maxflow(G,s,t)
    n = nv(G)
    f,x,r = shortestpath_augment_maxflow(G,s,t)
    mincut = min_cut(G,r,s)
    if mincut == []
        λ = maximum(map(i->G.weights[t-2,i],inneighbors(G,t-2)))
    else
        λ = maximum(map(e->G.weights[e[2],e[1]],mincut))
    end
    
    H = clamped_capacitygraph(G,λ)
    fλ,xλ,rλ =  shortestpath_augment_maxflow(H,s,t)
    mincut = min_cut(H,rλ,s)
    qλ = cap_freq(H, mincut, λ)
    while fλ < f
        λ += (f-fλ)/qλ
        H = clamped_capacitygraph(G,λ)
        fλ,xλ,rλ, =  shortestpath_augment_maxflow(H,s,t)
        mincut = min_cut(H,rλ,s)
        qλ = cap_freq(H, mincut, λ)
    end
    return fλ, xλ
end


#=
Procedure: cap_freq
Purpose: Finds number of edges in list with certain capacity in G
Parameters: G, a SimpleWeightedDiGraph
            lst, list of pairs of nodes represented directed edges
             λ, capacity to search for
Produces: freq, number of edges with capacity λ in lst (wrt G)
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function cap_freq(G, lst, λ)
    caps = map(e->G.weights[e[2],e[1]],lst)
    return length(findall(caps .== λ))
end


#=
Procedure: clamped_capacitygraph
Purpose: Clamp capacity of graph to λ
Parameters: G, a SimpleWeightedDiGraph
            λ, float, supremum of capacity in new graph
Produces: H, a SimpleWeightedDiGraph
Preconditions: none
Postconditions: for an edge (i,j) in G, its capacity in H is c[i,j] = min(c[i,j], λ) 
=#    
function clamped_capacitygraph(G,λ)
    n = nv(G)
    H = copy(G)
    for i in 1:n
        for j in outneighbors(H,i)
            H.weights[j,i] = min(H.weights[j,i], λ)
        end
    end
    return H
end

#----------------------------ADAPTIVE MAX FLOW-------------------------------#

#---------------EXACT VALUE-----------------#
#****NOTE: SINCE GLPK DOES NOT SUPPORT QUADRATIC OPTIMIZATION, THE NEXT

#=
Procedure: adaptive_maxflow
NOTE: SINCE GLPK DOES NOT SUPPORT QUADRATIC OPTIMIZATION, THIS PROCEDURE CURRENTLY DOES NOT WORK (THROWS HORRIFIC ERRORS)
Purpose: Finds maximum adaptive flow, which is defined in "Robust and Adaptive Network Flows" by Bertsimas et al.
Parameters: G, a SimpleWeightedDiGraph
            gam, max # of edges which globally fail in network at a time
            s, the source vertex
            t, the sink vertex
Produces: f, the maximum adaptive flow value from s to t
          x, the flow in the network (n x n matrix)
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function adaptive_maxflow(G,gam,s,t)
    n = nv(G)
    c = LightGraphs.weights(G) #capacity matrix
    edgl = map(e->(e.src,e.dst),edges(G)) #edge-list
    
    af = Model(GLPK.Optimizer)
    @variable(af, x[edgl] >= 0) #flow
    @variable(af, mu[edgl], Bin)
    @variable(af, y[edgl] >= 0)
    @variable(af, z >= 0)

    @constraint(af, [e=edgl], x[e] <= c[e[1],e[2]]) #flow cannot exceed capacity
    @constraint(af, [i=setdiff(vertices(G),[s,t])],
                sum([x[e] for e ∈ inedges(G,i)])-
                sum([x[e] for e ∈ outedges(G,i)]) == 0) #flow conservation for x

    @constraint(af, sum([mu[e] for e ∈ edgl]) <= gam) #max no of failures is gam
    @constraint(af, [e=edgl], y[e] <= x[e]) #flow after failure cannot exceed initial flow
    @constraint(af, [i=setdiff(vertices(G),[s,t])],
                sum([(1-mu[e])*y[e] for e ∈ inedges(G,i)])-
                sum([(1-mu[e])*y[e] for e ∈ outedges(G,i)]) == 0)
    
    @constraint(af, z-sum([(1-mu[e])*x[e] for e ∈ inedges(G,t)]) <= 0)
    @objective(af, Max, z)

    optimize!(af)
    @show value.(d)
    @show value.(th)
    return  objective_value(af), (value.(x)).data
end

#---------------APPROXIMATIONS-----------------#

#=
Procedure: adaptive_maxflow_approx
Purpose: Finds "approximate" maximum adaptive flow using LPP #39 from "Robust and Adaptive Network Flows" by Bertsimas et al.
Parameters: G, a SimpleWeightedDiGraph
            gam, max # of edges which globally fail in network at a time
            s, the source vertex
            t, the sink vertex
Produces: f, approximate maximum adaptive flow value from s to t
          x, flow in the network (n x n matrix)
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function adaptive_maxflow_approx(G,gam,s,t)
    n = nv(G)
    c = LightGraphs.weights(G) #capacity matrix
    edgl = setdiff(map(e->(e.src,e.dst),edges(G)),[(n-1,1),(n-2,n)]) #edge-list
    
    af = Model(GLPK.Optimizer)
    @variable(af, x[edgl] >= 0) #flow
    @variable(af, del >= 0)

    @constraint(af, [e=edgl], x[e] <= c[e[1],e[2]]) #flow cannot exceed capacity
    @constraint(af, [i=setdiff(vertices(G),[s,t,1,t-2])],
                sum([x[e] for e ∈ inedges(G,i)])-
                sum([x[e] for e ∈ outedges(G,i)]) == 0) #flow conservation for x
    @constraint(af, [e=edgl], x[e] <= del)
    @objective(af, Max, sum([x[e] for e ∈ inedges(G,t-2)]) - gam*del)

    optimize!(af)
    optx = value.(x)
    xflow = spzeros(n,n)
    for e in edgl
        xflow[e[1],e[2]] = optx[e]
    end
    return objective_value(af), xflow
end


#=
Procedure: adaptive_maxflow_sampling
Purpose: Approximates maximum adaptive flow using random sampling of failures 
Parameters: G, a SimpleWeightedDiGraph
            gam, max # of edges which globally fail in network at a time
            s, the source vertex
            t, the sink vertex
            Nsamples, number of samples to draw from weak compositions of each k<=gam
Produces: f, approximate maximum adaptive flow value from s to t
          x, flow in the network (n x n matrix)
Preconditions: * weight of an edge in G denotes its capacity
Postconditions: none 
=#
function adaptive_maxflow_sampling(G, gam, s, t, Nsamples, maxFrac=0.5)
    n = nv(G)
    m = ne(G)
    c = LightGraphs.weights(G) #capacity matrix
    edgl = map(e->(e.src,e.dst),edges(G))
    edge_indices = Dict(edgl[i]=>i for i in 1:m)
    edgl = setdiff(edgl,[(n-1,1),(n-2,n)]) #edge-list
    mu_es = []
    for g in 0:gam
        mu_es_g = weak_compositions(g, m, 10*Nsamples)
        l = length(mu_es_g)
        choices = sample(1:l, min(floor(Int,l*maxFrac), Nsamples), replace = false)
        map(i->push!(mu_es,mu_es_g[i]), choices)
    end
    
    af = Model(GLPK.Optimizer)
    @variable(af, x[edgl] >= 0) #flow
    @variable(af, y[edgl] >= 0)

    @constraint(af, [e=edgl], x[e] <= c[e[1],e[2]]) #flow cannot exceed capacity
    @constraint(af, [e=edgl,mu_e=mu_es], y[e] <= (1-mu_e[edge_indices[e]])*x[e]) #flow cannot exceed capacity for rerouted flow y
    @constraint(af, [i=setdiff(vertices(G),[s,t,1,t-2])],
                sum([x[e] for e ∈ inedges(G,i)])-
               sum([x[e] for e ∈ outedges(G,i)]) == 0) #flow conservation for x
    @constraint(af, [i=setdiff(vertices(G),[s,t,1,t-2])],
                sum([y[e] for e ∈ inedges(G,i)])-
                sum([y[e] for e ∈ outedges(G,i)]) == 0) #flow conservation for rerouted flow y

    @objective(af, Max, sum([y[e] for e ∈ inedges(G,t-2)]))

    optimize!(af)
    optx = value.(x)
    xflow = spzeros(n,n)
    for e in edgl
        xflow[e[1],e[2]] = optx[e]
    end
    xflow[t-2,t] = sum(xflow[:,n-2])
    xflow[t-1,1] = sum(xflow[1,:])
    return xflow[t-2,t], xflow
end


#-------------------------------EXPERIMENTS--------------------------------#

#----------------------OPTIMALITY COMPARISON------------------------#
#=
Procedure: all_maxflows
Purpose: Calculates classical solution, approx. adpative flow, sampled adaptive flow, and sampled robust flow to max flow problem
Parameters: netgenFile, a string, name of file with NETGEN networks
            nodeRange, int, range of network sizes to read from netgenFile 
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            gamma, max # of edges which globally fail in network at a time
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
Produces: 4 vectors, f_classical, f_adaptive_approx, f_adaptive_sampling, f_robust
          each vector represents the average value of the particular type of flow at each network size
Preconditions: none
Postconditions: none
=#
function all_maxflows(netgenFile, nodeRange, nodeStep, each, gamma, Nsamples)
    totalNodes = 1+nodeRange÷nodeStep
    f_classical = zeros(totalNodes)
    f_adaptive_approx = zeros(totalNodes)
    f_adaptive_sampling = zeros(totalNodes)
    f_robust = zeros(totalNodes)
    index = each
    for G in netgenNetworks(totalNodes*each, open(netgenFile))
        t = nv(G)
        s = t-1
        f_classical[index÷each] += shortestpath_augment_maxflow(G, s, t)[1]
        f_adaptive_approx[index÷each] += adaptive_maxflow_approx(G, gamma, s, t)[1]
        f_adaptive_sampling[index÷each] += adaptive_maxflow_sampling(G, gamma, s, t, Nsamples)[1]
        f_robust[index÷each] += robust_maxflow_globalapprox(G, gamma, s, t, Nsamples)[1]
        index += 1
    end
    return f_classical/each,  f_adaptive_approx/each,  f_adaptive_sampling/each, f_robust/each 
end

#=
Procedure: run_exper1
Purpose: Experiment to measure ratio of approx. adpative flow, sampled adaptive flow, and sampled robust flow values to max flow problem value
Parameters: netgenFile, a string, name of file with NETGEN networks
            minNodes, int, min number of nodes in generated networks
            maxNodes, int, max number of nodes in generated networks
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            gamma, max # of edges which globally fail in network at a time
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
            plotFile, string, name with which to save generated plot
Produces: list of 3 vectors, f_adaptive_ratio, f_adaptive_sampling_ratio, f_robust_ratio, each vector represents the average ratio of the value of particular type of flow to classical flow value, at each network size
Preconditions: none
Postconditions: plot is saved in plotFile
=#
function run_exper1(netgenFile, minNode, maxNode, nodeStep, each, gamma, Nsamples, plotFile)
    
    f = all_maxflows(netgenFile, maxNode - minNode, nodeStep, each, gamma, Nsamples)
    println(f)
    clf = f[1]
    totalNodes = length(clf)
    ratios = []
    p = plot(legend=:bottomright)
    nodes = collect(minNode:nodeStep:maxNode)
    labels = ["Adaptive flow using approximation alg", "Adaptive flow using sampling", "Robust flow using sampling"]
    for j in 1:length(f[2:end])
        rf = f[j+1]
        ratio = map(i->rf[i]/clf[i], 1:totalNodes)
        push!(ratios, ratio)
        plot!(p, nodes, ratio,label=labels[j])
    end
    title!(p,"Optimality ratios with classical flow (gamma=$gamma, N=$Nsamples)")
    plot(p, xlabel="Number of nodes")
    plot(p, ylabel="Ratio")
    savefig(p,plotFile)
    return ratios
end

#-------------GAMMA LIMIT FOR ROBUST FLOWS------------#

#=
Procedure: vanish_gamma
Purpose: Finds gamma at which adaptive and robust max flows fall below certain fraction of the classical value
Parameters: netgenFile, a string, name of file with NETGEN networks
            nodeRange, int, range of network sizes to read from netgenFile 
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
            minFlowFrac, minimum fraction of classical max flow value required 
Produces: 2 vectors, adaptive_vanish_gamma, robust_vanish_gamma
          each vector represents the average value of gamma at which that flow type falls below threshold minFlowFrac*(classical max flow) for each network size
Preconditions: 0 <= minFlowFrac <= 1
Postconditions: none
=#
function vanish_gamma(netgenFile, nodeRange, nodeStep, each, Nsamples, minFlowFrac)
    
    totalNodes = 1+nodeRange÷nodeStep
    robust_vanish_gamma = zeros(totalNodes)
    adaptive_vanish_gamma = zeros(totalNodes)
    index = each
    for G in netgenNetworks(totalNodes*each, open(netgenFile))
        if index%each == 0
            t = nv(G)
            s = t-1
            classical = shortestpath_augment_maxflow(G, s, t)[1]
            gamma = 0
            robustflow_gamma = classical
            while robustflow_gamma > minFlowFrac*classical
                gamma += 1
                robustflow_gamma = robust_maxflow_globalapprox(G, gamma, s, t, Nsamples)[1]
            end
            robust_vanish_gamma[index÷each] += gamma
            gamma = 0
            adaptiveflow_gamma = classical
            while adaptiveflow_gamma > minFlowFrac*classical
                gamma += 1
                adaptiveflow_gamma = adaptive_maxflow_sampling(G, gamma, s, t, Nsamples)[1] 
            end
            adaptive_vanish_gamma[index÷each] += gamma
        end
        index += 1
    end
    return adaptive_vanish_gamma, robust_vanish_gamma
end

#=
Procedure: run_exper2
Purpose: Experiment to measure gamma at which adaptive and robust max flows fall below certain fraction of the classical value
Parameters: netgenFile, a string, name of file with NETGEN networks
            minNode, int, min number of nodes in generated networks
            maxNode, int, max number of nodes in generated networks
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
            minFlowFrac, minimum fraction of classical max flow value required 
            plotFile, string, name with which to save generated plot
Produces: list of 2 vectors, adaptive_vanish_gamma, robust_vanish_gamma, each vector represents the average value of gamma at which particular type of flow falls below threhold minFlowFrac*(classical max flow) at each network size
Preconditions:  0 <= minFlowFrac <= 1
Postconditions: plot is saved in plotFile
=#
function run_exper2(netgenFile, minNode, maxNode, nodeStep, each, Nsamples, minFlowFrac, plotFile)
    
    vanish_gammas = vanish_gamma(netgenFile, maxNode-minNode, nodeStep, each, Nsamples, minFlowFrac)
    totalNodes = length(vanish_gammas[1])
    p = plot(legend=:bottomright)
    plot(p, size=(200,200))
    nodes = collect(minNode:nodeStep:maxNode)
    labels = ["Adaptive max flow", "Robust max flow"]
    for j in 1:length(vanish_gammas)
        plot!(p, nodes, vanish_gammas[j] ,label=labels[j])
    end
    title!(p,"Gamma at which max flows vanish (N=$Nsamples, alpha=$minFlowFrac)")
    plot(p, xlabel="Number of nodes")
    plot(p, ylabel="Gamma")
    savefig(p,plotFile)
    return vanish_gammas
end


#-----PERFORMANCE OF FLOWS WITH CAPACITY PERTURBATIONS-------#

#=
Procedure: perturbed_performance
Purpose: Measure performance of classical flow, adaptive flow, and robust flow under perturbation of edge capacities specified by perturbFrac
Parameters: netgenFile, a string, name of file with NETGEN networks
            nodeRange, int, range of network sizes to read from netgenFile 
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            gamma, max # of edges which globally fail in network at a time
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
            perturbFrac, modify each edge capacity c[i,j] by a perturbation chosen randomly from [-perturbFrac*c[i,j],perturbFrac*c[i,j]]   
Produces: list of 3 vectors, af_classical_error, f_adaptive_error, f_robust_error, each vector represents the average capacity violation in the perturbed graph for each network size
Preconditions:  0 < perturbFrac <= 1
Postconditions: none
=#
function perturbed_performance(netgenFile, nodeRange, nodeStep, each, gamma, Nsamples, perturbFrac)

    totalNodes = 1+nodeRange÷nodeStep
    f_classical_error = zeros(totalNodes)
    f_adaptive_error = zeros(totalNodes)
    f_robust_error = zeros(totalNodes)
    #f_newton_error = zeros(totalNodes)
    index = each
    for G in netgenNetworks(totalNodes*each, open(netgenFile))
        t = nv(G)
        s = t-1
        H = perturbed_cap_network(G, perturbFrac)
        
        f_c, x_c = shortestpath_augment_maxflow(G, s, t)[1:2]
        f_classical_error[index÷each] += flow_violation(H, x_c)
        
        f_a, x_a = adaptive_maxflow_sampling(G, gamma, s, t, Nsamples)
        f_adaptive_error[index÷each] += flow_violation(H, x_a)
        
        f_r, x_r = robust_maxflow_globalapprox(G, gamma, s, t, Nsamples)
        f_robust_error[index÷each] += flow_violation(H, x_r)

        #f_n, x_n = newton_maxflow(G, s, t)
        #f_newton_error[index÷each] += flow_violation(H, x_n)
        
        index += 1
        println(index)
    end
    return f_classical_error/each, f_adaptive_error/each, f_robust_error/each
end

#=
Procedure: run_exper3
Purpose: Experiment to measure performance of classical flow, adaptive flow, and robust flow under perturbation of edge capacities specified by perturbFrac
Parameters: netgenFile, a string, name of file with NETGEN networks
            minNode, int, min number of nodes in generated networks
            maxNode, int, max number of nodes in generated networks
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            gamma, max # of edges which globally fail in network at a time
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
            perturbFrac, modify each edge capacity c[i,j] by a perturbation chosen randomly from [-perturbFrac*c[i,j],perturbFrac*c[i,j]]
            plotFile, string, name with which to save generated plot
Produces: list of 3 vectors, af_classical_error, f_adaptive_error, f_robust_error, each vector represents the average capacity violation of flow in the perturbed graph for each network size
Preconditions:  0 < perturbFrac <= 1
Postconditions: plot is saved in plotFile
=#
function run_exper3(netgenFile, minNode, maxNode, nodeStep, each, gamma, Nsamples, perturbFrac, plotFile)
    
    errors = perturbed_performance(netgenFile, maxNode-minNode, nodeStep, each, gamma, Nsamples, perturbFrac)
    totalNodes = length(errors)
    p = plot(legend=:topright)
    plot(p, size=(200,200))
    nodes = collect(minNode:nodeStep:maxNode)
    labels = ["Classical max flow", "Adaptive max flow", "Robust max flow", "Newton max flow"]
    for j in 1:length(errors)
        plot!(p, nodes, errors[j] ,label=labels[j])
    end
    title!(p,"Perturbed capacity violation (N=$Nsamples, alpha=$perturbFrac)")
    plot(p, xlabel="Number of nodes")
    plot(p, ylabel="Capacity violation")
    savefig(p,plotFile)
    return errors
end

#=
Procedure: perturbed_cap_network
Purpose: Create new network with perturbed capacities 
Parameters: G, a SimpleWeightedDiGraph
            perturbFrac, modify each edge capacity c[i,j] by a perturbation chosen randomly with uniform probability from [-perturbFrac*c[i,j],perturbFrac*c[i,j]]   
Produces: H, a SimpleWeightedDiGraph with perturbed capacities
Preconditions: 0 < perturbFrac <= 1
Postconditions: none
=#
function perturbed_cap_network(G, perturbFrac)
    w = copy(LightGraphs.weights(G))
    for e in edges(G)
        w[e.src,e.dst] += w[e.src,e.dst]*rand(Uniform(-perturbFrac,0))
    end
    return SimpleWeightedDiGraph(w)
end

#=
Procedure: flow_violation
Purpose: Create new network with perturbed capacities 
Parameters: G, a SimpleWeightedDiGraph
            x, n x n matrix representing flow in G 
Produces: error, float value which is sum capacity constraint violations
Preconditions: none
Postconditions: none
=#
function flow_violation(G, x)
    error = 0.0
    for (s,d,f) in zip(findnz(x)...)
        if f > G.weights[d,s]
            error += (f - G.weights[d,s])
        end
    end
    return error
end


#--------------SIMILARITY OF FLOWS------------#

#=
Procedure: flow_similarity
Purpose: Measure pairwise similarity of classical flow, adaptive flow, and robust flow using Frobenius norm
Parameters: netgenFile, a string, name of file with NETGEN networks
            nodeRange, int, range of network sizes to read from netgenFile 
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            gamma, max # of edges which globally fail in network at a time
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
Produces: 3 vectors, sim_classical_robust, sim_robust_adaptive, sim_adaptive_classical representing average similarity of each pair of flows for each network size
Preconditions:  none
Postconditions: none
=#
function flow_similarity(netgenFile, nodeRange, nodeStep, each, gamma, Nsamples)

    totalNodes = 1+nodeRange÷nodeStep
    sim_classical_robust = zeros(totalNodes)
    sim_robust_adaptive = zeros(totalNodes)
    sim_adaptive_classical = zeros(totalNodes)
    index = each
    for G in netgenNetworks(totalNodes*each, open(netgenFile))
        t = nv(G)
        s = t-1
        
        f_c, x_c = shortestpath_augment_maxflow(G, s, t)[1:2]
        f_r, x_r = robust_maxflow_globalapprox(G, gamma, s, t, Nsamples)
        f_a, x_a = adaptive_maxflow_sampling(G, gamma, s, t, Nsamples)

        sim_classical_robust[index÷each] += norm(x_c-x_r, 2)
        sim_robust_adaptive[index÷each] += norm(x_r-x_a, 2)
        sim_adaptive_classical[index÷each] += norm(x_a-x_c, 2)
        
        index += 1
    end
    return sim_classical_robust/each, sim_robust_adaptive/each, sim_adaptive_classical/each
end

#=
Procedure: run_exper4
Purpose: Experiment to measure pairwise similarity of classical flow, adaptive flow, and robust flow using Frobenius norm
Parameters: netgenFile, a string, name of file with NETGEN networks
            minNode, int, min number of nodes in generated networks
            maxNode, int, max number of nodes in generated networks
            nodeStep, int, steps which networks were generated in [minNodes,maxNodes]
            each, int, number of networks for each network size
            gamma, max # of edges which globally fail in network at a time
            Nsamples, number of samples to draw from weak compositions to approximate adpative flow and robust flow
            plotFile, string, name with which to save generated plot
Produces: list of 3 vectors, af_classical_error, f_adaptive_error, f_robust_error, each vector represents the average capacity violation in the perturbed graph for each network size
Preconditions:  none
Postconditions: plot is saved in plotFile
=#
function run_exper4(netgenFile, minNode, maxNode, nodeStep, each, gamma, Nsamples, plotFile)
    
    sims = flow_similarity(netgenFile, maxNode-minNode, nodeStep, each, gamma, Nsamples)
    totalNodes = length(sims)
    p = plot(legend=:topright)
    plot(p, size=(200,200))
    nodes = collect(minNode:nodeStep:maxNode)
    labels = ["Classical and robust", "Robust and adaptive", "Adaptive and classical"]
    for j in 1:length(sims)
        plot!(p, nodes, sims[j] ,label=labels[j])
    end
    title!(p,"Frobenius similarity of different flows")
    plot(p, xlabel="Number of nodes")
    plot(p, ylabel="Similarity")
    savefig(p,plotFile)
    return sims
end
    

#-------------------------------FLOW PLOTTING--------------------------------#

#=
Procedure: plot_flow
Purpose: Plots G with edge labels (capacity, flow)
Parameters: G, a SimpleWeightedDiGraph
            x, network flow in G (either n x n matrix or vector of flows)
            file, string for filename
Produces: none
Preconditions: if x is vector, flows should be in canonical sorted order
Postconditions: saves plot as pdf with name file ("maxflow.pdf" by default) 
=#
function plot_flow(G,x,file="maxflow.pdf")
    edge_cap =  filter(w->w!=0,vec(LightGraphs.weights(G)'))
    x2 = x
    dim = size(x) 
    if length(dim) == 2 && dim[1]==dim[2] #if x is a matrix
        x2 = map(e->x[e.src,e.dst],edges(G))
    end
    pairs = collect(zip(edge_cap,x2))
    draw(PDF(file, 16cm, 16cm),gplot(G,nodelabel=1:nv(G),edgelabel=pairs))
end
