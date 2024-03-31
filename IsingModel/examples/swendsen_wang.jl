using CairoMakie, DelimitedFiles

filename = joinpath(@__DIR__, "sw.dat")

@info "Plot the data..."
data = readdlm(filename)
fig = Figure()
ax = Axis(fig[1, 1], xlabel="time")
legends = [L"energy/spin", L"(energy/spin)^2", L"|m|", L"m^2", L"m^4"]

for i=1:5
    lines!(ax, data[:, i], label=legends[i])
end
axislegend(ax)
fig

filename = joinpath(@__DIR__, "swmdata.png")
save(filename, fig, px_per_unit=2)
@info "The plot is saved to: `$filename`."


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



# Writes the bin averages to the file res.dat
#--------------------------------------------
 function writedata(mstp,mdata)
    mdata.=mdata./mstp
    f=open("res.dat","a")
       println(f,mdata[1],"  ",mdata[2],"  ",mdata[3],"  ",mdata[4],"  ",mdata[5])
    close(f)
    mdata.=0.
 end


ll = 100
tt = 2.0
hh = 0.0
istp = 1000
mstp = 1000
bins = 100

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

# original time approximately equals to 30s