 function processdata()
    dd=zeros(Float64,5)   
    av=zeros(Float64,2,5)
    n=0
    file=open("res.dat","r")
    while !eof(file)
       dd[1]=parse(Float64,readuntil(file,"  "))
       dd[2]=parse(Float64,readuntil(file,"  "))
       dd[3]=parse(Float64,readuntil(file,"  "))
       dd[4]=parse(Float64,readuntil(file,"  "))
       dd[5]=parse(Float64,readline(file))
       dd[2]=dd[2]-dd[1]^2
       dd[5]=dd[5]/dd[4]^2
       @. av[1,:]=av[1,:]+dd[:]  
       @. av[2,:]=av[2,:]+dd[:]^2     
       n=n+1
    end
    close(file)
    println(n)
    @. av[:,:] = av[:,:]/n
    @. av[2,:] = abs((av[2,:]-av[1,:]^2)/(n-1))^0.5
    return av
 end

 av = processdata()

 file=open("read.in","r")
    ll=parse(Int64,readline(file))      # Lattice size (length)
    tt=parse(Float64,readline(file))    # Temperature (J=1)
 close(file)

 av[:,2].=av[:,2].*(ll/tt)^2

 f1=open("e.dat","w")
 f2=open("m.dat","w")
 f3=open("q.dat","w")
 println(f1,av[1,1],"  ",av[2,1],"  ",av[1,2],"  ",av[2,2])
 println(f2,av[1,3],"  ",av[2,3],"  ",av[1,4],"  ",av[2,4])
 println(f3,av[1,5],"  ",av[2,5])
 close(f1)
 close(f2)
 close(f3)
