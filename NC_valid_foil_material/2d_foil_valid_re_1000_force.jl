
using WaterLily
using StaticArrays
using Plots
using WaterLily,StaticArrays
using ParametricBodies
using DelimitedFiles,CSV   

using Plots; gr()
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
   WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
   contour!(sim.flow.σ[R]'|>Array;levels,lines)

   # body 
   data = sim.flow.σ[R]' |> Array

end

function flood(f::Array;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=0, levels=levels, color=cfill, clims = clims, 
        aspect_ratio=:equal; kv...)
end
# ----------------  力的计算 ----------------
function pressure_force(sim;t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
   sim.flow.f .= zero(eltype(sim.flow.p))
   @WaterLily.loop sim.flow.f[I,:] .= sim.flow.p[I]*WaterLily.nds(sim.body,loc(0,I,T),t) over I ∈ inside(sim.flow.p)
   sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.u)-1))[:] |> Array
end

function viscous_force(sim;t=WaterLily.time(sim.flow),T=promote_type(Float64,eltype(sim.flow.p)))
  sim.flow.f .= zero(eltype(sim.flow.p))
  @WaterLily.loop sim.flow.f[I,:] .= -2sim.flow.ν*WaterLily.S(I,sim.flow.u)*WaterLily.nds(sim.body,loc(0,I,T),t) over I ∈ WaterLily.inside_u(sim.flow.u)
  sum(T,sim.flow.f,dims=ntuple(i->i,ndims(sim.flow.u)-1))[:] |> Array
end

function make_sim(;L=48,Re=1e3,St=0.3,U=1,T=Float32,mem=Array,aoa)
   # Map from simulation coordinate x to surface coordinate ξ
   nose,pivot = SA[1L,3.5L],SA[0.f0L,0]
   #前后攻角
   fist_aoa = aoa
   ######input parameter
   Dis = 2L #相距
   ###得到全局坐标
   function map(x,t)

       back = false      # back body?
       h = back ? 0 : 0 # y shift
       S = back ? Dis : 0           # horizontal shift
       θ1 = back ? T(0/180*π) : T(fist_aoa/180*π) 
       R = SA[cos(θ1) -sin(θ1); sin(θ1) cos(θ1)]
       h = SA[S,h]#h₀*sin(ω*t)]
       ξ = R*(x-nose-h-pivot)+pivot # move to origin and align with x-axis
       return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
   end

   # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
   NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
   foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]
   

   body = HashedBody(foil,(0,1);map,T,mem)


   Simulation((8L,7L),(U,0),L;ν=U*L/Re,body,T,mem)
end
function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
   remeasure=true,plotbody=true,kv...)
   t₀ = round(sim_time(sim))
   @time @gif for tᵢ in range(t₀,t₀+duration;step)
      sim_step!(sim,tᵢ;remeasure)
      @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
      flood(sim.flow.σ[R]|>Array; kv...)
      body_plot!(sim)

      verbose && println("tU/L=",round(tᵢ,digits=4),
      ", Δt=",round(sim.flow.Δt[end],digits=3))

   end


end
function get_forces!(sim,t)

   sim_step!(sim,t,remeasure=false)
  
   force = pressure_force(sim) + viscous_force(sim)
   println("t cd cl",force./(0.5sim.L*sim.U^2))
   force./(0.5sim.L*sim.U^2) # scale the forces!
end
# intialize and run

aoa =[0,8,10,15,20]

for aoa_i in aoa
   sim = make_sim(aoa=aoa_i)
   #sim_gif!(sim,duration=1,clims=(-10,10),plotbody=true)

   # Simulate through the time range and get forces
   ts = 1:0.1:50 # time scale is sim.L/sim.U
   forces = [get_forces!(sim,t) for t in ts];

   # 1. 先把数据整理成一个二维矩阵：列依次为 t, drag, lift
   data = hcat(ts, first.(forces), last.(forces))
   header = "time,drag,lift"
   # 2. 写入 CSV（以逗号为分隔符）
   dir  = "valid_foil/Force"                
   mkpath(dir)                 
   
   filename = joinpath(dir, "forces_aoa$(aoa_i).csv")
   open(filename, "w") do io
      println(io, header)           
      writedlm(io, data, ',')       # data
  end
   @info "✓ 保存 $(filename)"
  
end