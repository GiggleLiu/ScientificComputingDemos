using Test, IsingModel, DelimitedFiles

@testset "lattice" begin
    function original_lattice(ll,nn,neighbor,bondspin,spinbond)
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
    neighbor, bondspin, spinbond = zeros(Int, 4, 9), zeros(Int, 2, 18), zeros(Int, 4, 9)
    original_lattice(3, 9, neighbor, bondspin, spinbond)
    neighbor_ = IsingModel.lattice(3)
    bondspin_, spinbond_ = IsingModel.spinbondmap(neighbor_)
    @test neighbor == neighbor_
    @test bondspin == bondspin_
    @test spinbond == spinbond_
end