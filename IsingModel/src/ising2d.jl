
# Constructs a list neigh[1:4,1:nn] of neighbors of each site

 function lattice(ll)
    neigh=Array{Int}(undef,4,ll*ll)
    for s0=1:ll*ll
       x0=mod(s0-1,ll)
       y0=div(s0-1,ll)
       x1=mod(x0+1,ll)
       x2=mod(x0-1+ll,ll)
       y1=mod(y0+1,ll)
       y2=mod(y0-1+ll,ll)
       neigh[1,s0]=1+x1+y0*ll
       neigh[2,s0]=1+x0+y1*ll  
       neigh[3,s0]=1+x2+y0*ll
       neigh[4,s0]=1+x0+y2*ll
    end
    return neigh
 end

lattice(5)

# Constructs the initial random spin configuration

 function initspin(nn::Int)
    spin=Array{Int32}(undef,nn)
    for i=1:nn
       spin[i]=rand(-1:2:1)
    end
    return spin
 end

# Performs one MC sweep. This version computes the neighbors on the fly.

 function mcstep1(ll::Int,pflp,spin)
    nn::Int=ll*ll
    for i=1:nn
       s::Int=rand(1:nn)
       x::Int=mod(s-1,ll)
       y::Int=div(s-1,ll)
       s1::Int=spin[1+mod(x+1,ll)+y*ll]
       s2::Int=spin[1+x+mod(y+1,ll)*ll]
       s3::Int=spin[1+mod(x-1,ll)+y*ll]
       s4::Int=spin[1+x+mod(y-1,ll)*ll]
       if rand()<pflp[spin[s]+2,s1+s2+s3+s4+5] 
          spin[s]=-spin[s]
       end
    end    
 end

# Performs one MC sweep. This version uses the list of neighbors 

 function mcstep2(ll::Int,neigh,pflp,spin)
    nn::Int=ll*ll
    for i=1:nn
       s::Int=rand(1:nn)
       s1::Int=spin[neigh[1,s]]
       s2::Int=spin[neigh[2,s]]
       s3::Int=spin[neigh[3,s]]
       s4::Int=spin[neigh[4,s]]
       if rand()<pflp[spin[s]+2,s1+s2+s3+s4+5] 
          spin[s]=-spin[s]
       end
    end    
 end

# Measures physical quantities and accumulates them in mdata
# mdata[1] = energy/spin
# mdata[2] = (energy/spin)^2
# mdata[3] = |m|
# mdata[4] = m^2
# mdata[5] = m^4

function measure(ll::Int,spin,mdata)
   e::Int64=0
   for y=0:ll-1
      for x=0:ll-1
         e=e-spin[1+x+y*ll]*(spin[1+mod(x+1,ll)+y*ll]+spin[1+x+mod(y+1,ll)*ll])
      end
   end
   m::Int64=sum(spin)
   n::Int64=ll*ll
   mdata[1] += e/n
   mdata[2] += (e/n)^2
   mdata[3] += abs(m/n)
   mdata[4] += (m/n)^2
   mdata[5] += (m/n)^4
end

# Writes the bin averages to the file res.dat, writes a message to 'log.log'

function writedata(j,mstp,mdata)
    mdata.=mdata./mstp
    f=open("res.dat","a")
       println(f,mdata[1],"  ",mdata[2],"  ",mdata[3],"  ",mdata[4],"  ",mdata[5])
    close(f)
    mdata.=0
    f=open("log.log","w")
       println(f,"Completed bin ",j)
    close(f)
end

# an example for testing
ll = 100
tt = 2.0
hh = 0.0
istp = 1000
mstp = 1000
bins = 100

nn=ll^2
neigh=lattice(ll)
spin=initspin(nn)

# Constructs the acceptance probabilities

 pflp=Array{Float64}(undef,3,9)
 for i=-4:2:4
    pflp[1,i+5]=exp(+2*(i+hh)/tt)
    pflp[3,i+5]=exp(-2*(i+hh)/tt)
 end

@time begin

 for i=1:istp
    mcstep1(ll,pflp,spin)    
 end
 mdata=zeros(Float64,5)
 for j=1:bins 
    for i=1:mstp
       mcstep1(ll,pflp,spin)    
#       mcstep2(ll,neigh,pflp,spin)    
       measure(ll,spin,mdata)
    end
    writedata(j,mstp,mdata)
 end

end
