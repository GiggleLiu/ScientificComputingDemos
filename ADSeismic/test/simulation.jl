using ADSeismic, Test

@testset "loss" begin
     nx = 100
     ny = 100
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=2000)

     c = 1000*ones(param.NX+2, param.NY+2)
     src = (param.NX÷2, param.NY÷2)
     srcv = Ricker(param, 100.0, 500.0)
     tu = solve(param, src, srcv, c)
     tu2 = ADSeismic.solve_final(param, src, srcv, c)
     @test tu[:,:,end] ≈ tu2
     loss = sum(tu .^ 2)
     @test loss ≈ 10.931466822080788
end
