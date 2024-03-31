struct SwendsenWangModel{RT} <: AbstractSpinModel
    l::Int
    h::RT
    beta::RT
    pflp::NTuple{10, RT}
    neigh::Matrix{Int}
    bondspin::Matrix{Int}
    spinbond::Matrix{Int}
end
function SwendsenWangModel(l::Int, h::RT, beta::RT) where RT
    pflp = ([exp(-2*s*(i + h) * beta) for s=-1:2:1, i in -4:2:4]...,)
    neigh = lattice(l)
    bondspin, spinbond = spinbondmap(neigh)
    SwendsenWangModel(l, h, beta, pflp, neigh, bondspin, spinbond)
end

# Constructs the tables corresponding to the lattice structure (here 2D square)
# - the sites (spins) are labeled 1,...,N. The bonds are labeled 1,...,2N.
# - neighbor[i,s] = i:th neighbor sote of site s (i=1,2,3,4)
# - bondspin[i,b] = i:th site connected to bond b (i=1,2)
# - spinbond[i,s] = i:th bond connected to spin s (i=1,2,3,4)
#---------------------------------------------------------------------------------
function spinbondmap(neighbor)
    nn = size(neighbor, 2)
    bondspin = zeros(Int, 2, 2*nn)
    spinbond = zeros(Int, 4, nn)
    for s0 = 1:nn
        # map bonds to spin
        bondspin[1, s0] = s0
        bondspin[2, s0] = neighbor[1, s0]
        bondspin[1, nn+s0] = s0
        bondspin[2, nn+s0] = neighbor[2, s0]
        # map spins to bonds
        spinbond[1, s0] = s0
        spinbond[2, s0] = nn + s0
        spinbond[3, neighbor[1, s0]] = s0
        spinbond[4, neighbor[2, s0]] = nn + s0
    end
    return bondspin, spinbond
end

# Generates a valid bond configuration, given a spin configuration
#------------------------------------------------------------------
function castbonds(prob::Real, bondspin::AbstractMatrix, spin)    
    bond = zeros(Bool, size(bondspin, 2))
    for b in eachindex(bond)
        if spin[bondspin[1,b]] == spin[bondspin[2,b]]  # parallel spins
            bond[b] = rand() < prob
        else
            bond[b]=false
        end
    end 
    return bond
end

# Constructs all the clusters and flips each of them with probability 1/2
#-------------------------------------------------------------------------
function flipclusters(spin,bond,neighbor,spinbond,cflag,cstack)
   cflag.=true
   cseed::Int64=1
   nstack::Int64=1

   while cseed > 0    # construct clusters until all sites visited (then cseed=0)
      nstack=1
      cstack[1]=cseed
      cflag[cseed]=false

      if rand(0:1) == 1 # flip cluster 
         spin[cseed]=-spin[cseed]
         while nstack > 0
            s0::Int64=cstack[nstack]
            nstack=nstack-1
            for i=1:4
               s1::Int64=neighbor[i,s0]
               if bond[spinbond[i,s0]] && cflag[s1]
                  cflag[s1]=false
                  nstack=nstack+1
                  cstack[nstack]=s1
                  spin[s1]=-spin[s1]
               end 
            end
         end

      else # do not flip cluster
         while nstack > 0
            s0::Int64=cstack[nstack]
            nstack=nstack-1
            for i=1:4
               s1::Int64=neighbor[i,s0]
               if bond[spinbond[i,s0]] && cflag[s1]
                  cflag[s1]=false
                  nstack=nstack+1
                  cstack[nstack]=s1
               end 
            end
         end
      end

      dseed=cseed
      for i=cseed+1:length(spin)   # loop until a not visited site is found; the next cluster ssed
         if cflag[i]
            cseed=i
            break
         end
      end
      if dseed==cseed   # if no not-visited site is found, cseed=0, which signals completion 
         cseed=0
      end
   end
   return nothing
end