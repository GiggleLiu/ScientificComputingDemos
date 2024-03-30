# Constructs the tables corresponding to the lattice structure (here 2D square)
# - the sites (spins) are labeled 1,...,N. The bonds are labeled 1,...,2N.
# - neighbor[i,s] = i:th neighbor sote of site s (i=1,2,3,4)
# - bondspin[i,b] = i:th site connected to bond b (i=1,2)
# - spinbond[i,s] = i:th bond connected to spin s (i=1,2,3,4)
#---------------------------------------------------------------------------------
function lattice(ll,nn,neighbor,bondspin,spinbond)
   for s0=1:nn
      x0=mod(s0-1,ll)
      y0=div(s0-1,ll)
      x1=mod(x0+1,ll)
      x2=mod(x0-1+ll,ll)
      y1=mod(y0+1,ll)
      y2=mod(y0-1+ll,ll)
      s1=1+x1+y0*ll
      s2=1+x0+y1*ll  
      s3=1+x2+y0*ll
      s4=1+x0+y2*ll
      neighbor[1,s0]=s1
      neighbor[2,s0]=s2
      neighbor[3,s0]=s3
      neighbor[4,s0]=s4
      bondspin[1,s0]=s0
      bondspin[2,s0]=s1
      bondspin[1,s0+nn]=s0
      bondspin[2,s0+nn]=s2
      spinbond[1,s0]=s0
      spinbond[2,s0]=s0+nn
      spinbond[3,s1]=s0
      spinbond[4,s2]=s0+nn
   end
   return nothing
end

# generates a random spin configuration
#---------------------------------------
function initspin(spin)
   for i=1:length(spin)
      spin[i]=rand(-1:2:1)
   end
   return nothing   
end

# Generates a valid bond configuration, given a spin configuration
#------------------------------------------------------------------
function castbonds(prob,bondspin,spin,bond)    
   for b=1:length(bond)
      if spin[bondspin[1,b]]==spin[bondspin[2,b]]
         if rand()<prob
            bond[b]=true
         else
            bond[b]=false 
         end 
      else
         bond[b]=false
      end
   end 
   return nothing
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

# Measures physical observables
#------------------------------
function measure(ll,spin,mdata)
   e::Int64=0
   s0::Int64=0
   s1::Int64=0
   s2::Int64=0
   for y=0:ll-1
      for x=0:ll-1
         e=e-spin[1+x+y*ll]*(spin[1+mod(x+1,ll)+y*ll]+spin[1+x+mod(y+1,ll)*ll])
      end
   end
   n::Int64=ll*ll
   m::Int64=sum(spin)
   mdata[1] += e/n
   mdata[2] += (e/n)^2
   mdata[3] += abs(m/n)
   mdata[4] += (m/n)^2
   mdata[5] += (m/n)^4
end

# Writes the bin averages to the file res.dat
#--------------------------------------------
 function writedata(mstp,mdata)
    mdata.=mdata./mstp
    f=open("res.dat","a")
       println(f,mdata[1],"  ",mdata[2],"  ",mdata[3],"  ",mdata[4],"  ",mdata[5])
    close(f)
    mdata.=0.
 end

f=open("read.in","r")
   ll=parse(Int64,readline(f))    # lattice length
   tt=parse(Float64,readline(f))  # temperature
   hh=parse(Float64,readline(f))  # magnetic field hh=0 here, used for consistency with ising2d.jl
   istp=parse(Int64,readline(f))  # steps for equilibration
   mstp=parse(Int64,readline(f))  # steps for measurements
   bins=parse(Int64,readline(f))  # number of bins
close(f)

const nn=ll^2
const nb=2*nn
const prob=1-exp(-2/tt)

spin=Array{Int64}(undef,nn)
bond=Array{Bool}(undef,nb)
cflag=Array{Bool}(undef,nn)
cstck=Array{Int64}(undef,nn)

neighbor=Array{Int64}(undef,4,nn)
bondspin=Array{Int64}(undef,2,nb)
spinbond=Array{Int64}(undef,4,nn)
lattice(ll,nn,neighbor,bondspin,spinbond)

initspin(spin)

@time begin

for i=1:istp
   castbonds(prob,bondspin,spin,bond)    
   flipclusters(spin,bond,neighbor,spinbond,cflag,cstck)
end

mdata=zeros(Float64,5)
for j=1:bins
   for i=1:mstp
      castbonds(prob,bondspin,spin,bond)    
      flipclusters(spin,bond,neighbor,spinbond,cflag,cstck)
      measure(ll,spin,mdata)
   end
   writedata(mstp,mdata)
end

end
 
