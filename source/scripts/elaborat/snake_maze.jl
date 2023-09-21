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

image_path = "elaborat/mazes/snake_maze.png"
target_width = 175
target_height = 100
coordinates = get_black_pixel_coordinates_scaled(image_path, target_width, target_height)
cartesian_indices = [CartesianIndex(coord) for coord in coordinates]
w = target_width+1
h = target_height+1
depleting = true



function get_cfg(depleting,rnd_seed,production)
    cfg = Dict(
        :size => (w, h),
        :torus => (true, true),
        :seed => rnd_seed,

        :sim => nothing,

        :vaxes_rule_steps => 50,

        :barrier => @d(
            :vax => :wall,
            :pos => cartesian_indices
        ),

        :vaxes => @d(
            :wall => @d(:D => 0,
                        :d => 0,
                        :rd => 0),
            :e => @d(:D => 0.4,
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
                :k => 0.01,
                :w => 10.0,
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
                :receptors => @d(:re => 3.0),
                :state => @d(:cum_state => :re, :cum_state_weight => 1.0, :resting_time => 0),
                :init_pos => (25,50)
            ),
            @d(
                :receptors => @d(:x => 1.0),
                :state => @d(:cum_state => :e, :cum_state_weight => 1.0, :resting_time => 0),
                :init_pos => (160,15)
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

        :runtime => @d(:show_sim => true)
    )
end



function sim_test(depleting,rnd_seed,production)
    sim_desc = init_sim(get_cfg(depleting,rnd_seed,production))
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

df_sim = DataFrame(seeds=[],dep=[],steps=[],produ=[])
seeds=[]
dep=[]
steps=[]
produ=[]
for y in [10000]
    production = y
    println("Produkce látky: "*string(production))
    for i in (1)
        rnd_seed = rand(1:9999)
        time = sim_test(true,rnd_seed,production)
        df_sim = DataFrame(seeds=[],dep=[],steps=[],produ=[])
        push!(df_sim,[rnd_seed,true,time,production])
        CSV.write("snake_data.csv", df_sim, append=true, bufsize=2^28)
        println("Simulace číslo "*string(i)*"-true proběhla úspěšně")
        println("počet kroků: "*string(time))
        println(rnd_seed)
        time = sim_test(false,rnd_seed,production)
        df_sim = DataFrame(seeds=[],dep=[],steps=[],produ=[])
        push!(df_sim,[rnd_seed,false,time,production])
        CSV.write("snake_data.csv", df_sim, append=true, bufsize=2^28)
        println("Simulace číslo "*string(i)*"-false proběhla úspěšně")
        println("počet kroků: "*string(time))
        println(rnd_seed)
        
    end
end


#=
seed, time = sim_test(true,1853,1000)
println(time)
=#