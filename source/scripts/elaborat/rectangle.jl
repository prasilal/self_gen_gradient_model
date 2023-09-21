using Images
using DataFrames
using CSV

function get_black_pixel_coordinates_scaled(image_path, target_width, target_height)

    img_org = load(image_path)
    img = Gray.(img_org)

    height, width = size(img)

    black_pixel_coordinates = Set{Tuple{Int, Int}}()

    x_scale = target_width / width
    y_scale = target_height / height
    

    for y in 1:height
        for x in 1:width

            pixel_value = img[y, x]

            if pixel_value <= Gray(0.5)

                scaled_x = round(Int, x * x_scale)
                scaled_y = round(Int, y * y_scale)

                push!(black_pixel_coordinates, (scaled_x+1, scaled_y+1))
            end
        end
    end

    return collect(black_pixel_coordinates)
end

image_path = "elaborat/mazes/rectangle.png"
target_width = 100
target_height = 58
coordinates = get_black_pixel_coordinates_scaled(image_path, target_width, target_height)
cartesian_indices = [CartesianIndex(coord) for coord in coordinates]
w = target_width+1
h = target_height+1
depleting = true



function get_cfg(depleting,rnd_seed,production,konst_k,num_rec,vrs,diffu)
    cfg = Dict(
        :size => (w, h),
        :torus => (true, true),
        :seed => rnd_seed,

        :sim => nothing,

        :vaxes_rule_steps => vrs,

        :barrier => @d(
            :vax => :wall,
            :pos => cartesian_indices
        ),

        :vaxes => @d(
            :wall => @d(:D => 0,
                        :d => 0,
                        :rd => 0),
            :e => @d(:D => diffu,
                    :d => 0.0,
                    :rd => 0,
                    :show => true
                    ),
            :re => @d(:D => 0,
                    :d => 0,
                    :rd => 0),
            :x => @d(:D => 0,
                    :d => 0,
                    :rd => 0)
        ),

        :reactions => [
            @d(
                :react => [(1, :re), (1, :e)],
                :prod => [(1, :re)],
                :k => konst_k,
                :w => 5.0,
                :r_absorb => depleting,
            ),
            @d(
                :react => [(1, :re), (1, :x)],
                :prod => [(1, :re)],
                :k => 0.001,
                :w => 100.0,
                :r_absorb => false,
            )
        ],

        :cells => [
            @d(
                :receptors => @d(:re => num_rec),
                :state => @d(:cum_state => :re, :cum_state_weight => 1.0, :resting_time => 0),
                :init_pos => (5,28)
            ),
            @d(
                :receptors => @d(:x => 1.0),
                :state => @d(:cum_state => :e, :cum_state_weight => 1.0, :resting_time => 0),
                :init_pos => (95,28)
            )
        ],

        :rule_graph => @d(
            :min_weight => 0.0,
            :resting_time => 1,
            :zg => @d(
                :wall => [],
                :e => [(:e, :prod_v, production),(:e, :adhesion, 100), (:e, :volume, 50, 30),
                        (:e, :perimeter, 2, 70)],
                :re => [(:re, :adhesion, 100), (:re, :volume, 50, 30),
                        (:re, :perimeter, 2, 70),
                        (:e, :move, 12000)],
                :x => [(:x, :adhesion, 100), (:x, :volume, 50, 30),
                        (:x, :perimeter, 2, 70)]
            ),
            :cpm => @d(
                :T => 20,
                :other_adhesion => 20,
            )
        ),

        :runtime => @d(:show_sim => false)
    )
end



function sim_test(depleting,rnd_seed,production,konst_k,num_rec,vrs,diffu)
    sim_desc = init_sim(get_cfg(depleting,rnd_seed,production,konst_k,num_rec,vrs,diffu))
    state = sim_desc[:sim].cells[2].state[:cum_state]

    time = 0
    while (sim_desc[:sim].cells[2].state[:cum_state]==state)
        simulate(sim_desc, num_of_steps = 1)
        time += 1
    end
    end_state = sim_desc[:sim].cells[2].state[:cum_state]
    time
end


#Generování dat do csv

#seed, depletion, number of steps, prod_v, k, number of receptors, vaxes_rule_steps, diffusion

df_sim = DataFrame(seeds=[],dep=[],steps=[],produ=[],konsts_k=[],nums_rec=[],vrss=[],diffs=[])
seeds=[]
dep=[]
steps=[]
produ=[]
konsts_k=[]
nums_rec=[]
vrss=[]
diffs=[]

#seed, depletion, number of steps, prod_v, k, number of receptors, vaxes_rule_steps, diffusion

diffu=0.4
konst_k=0.01
num_rec=3.0
vrs=50
production=10000
for y in (10,20,30,40,50)
    vrs = y
    println("Aktuální verze parametru: "*string(y))
    for i in (1:3)
        rnd_seed = rand(1:9999)
        time = sim_test(true,rnd_seed,production,konst_k,num_rec,vrs,diffu)
        push!(df_sim,[rnd_seed,true,time,production,konst_k,num_rec,vrs,diffu])
        println("Simulace číslo "*string(i)*"-true proběhla úspěšně")
        println("počet kroků: "*string(time))
        println(rnd_seed)
        time = sim_test(false,rnd_seed,production,konst_k,num_rec,vrs,diffu)
        push!(df_sim,[rnd_seed,false,time,production,konst_k,num_rec,vrs,diffu])
        println("Simulace číslo "*string(i)*"-false proběhla úspěšně")
        println("počet kroků: "*string(time))
        println(rnd_seed)
    end
end
CSV.write("elaborat/rectangle_vrs.csv", df_sim, append=true, bufsize=2^28)


#=
seed, time = sim_test(true,1853,1000)
println(time)
=#