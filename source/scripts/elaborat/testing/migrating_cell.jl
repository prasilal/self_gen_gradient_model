cfg = Dict(
    :size => (100, 100),
    :torus => (true, true),
    :seed => 1234,

    :sim => nothing,

    :vaxes_rule_steps => 10,

    :vaxes => @d(
        :v => @d(:D => 0,
                  :d => 0.0001,
                  :rd => 0.00001
                 ),
        :a => @d(:D => 0.1,
                  :d => 0.0001,
                  :rd => 0.00001,
                  :show => true,
                  :init => (10:20, 80:90, 5)
                 ),
        :va => @d(:D => 0,
                  :d => 0.1, #0.001
                  :rd => 0.0001)

    ),

    :reactions => [
        @d(
            :react => [(1, :v), (1, :a)],
            :prod => [(1, :va)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
      @d(
        :receptors => @d(:v => 0.05),
        :state => @d(:cum_state => :v, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (50, 50)
      )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :v => [(:v, :prod_r, 0.001), (:v, :adhesion, 100), (:v, :volume, 50, 50), 
                   (:v, :perimeter, 2, 85), (:a, :move, 12000), (:v, :activity, 200, 30)],
            :a => [],
            :va => []
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    ),

    :runtime => @d()
)

sim_desc = init_sim(cfg)

simulate(sim_desc, num_of_steps = 500)



output=(print_zg(
    (
        :v => [(:v, :prod_r, 0.001), (:y, :prod_r, 0.02),
                    (:v, :adhesion, 100), (:v, :volume, 50, 50), (:v, :perimeter, 2, 85),
                    (:a, :move, 12000), (:v, :activity, 200, 30)],

            :a => [(:z, :prod_r, 0.0001), (:a, :prod_v, 0.1),
                     (:a, :adhesion, 100), (:a, :volume, 50, 50), (:a, :perimeter, 2, 85)],

            :x => [(:x, :prod_r, 0.001), (:x, :adhesion, 100), (:x, :volume, 50, 50), (:x, :perimeter, 2, 45)],

            :y => [(:y, :adhesion, 100), (:y, :volume, 50, 50), (:y, :perimeter, 2, 85), (:y, :resting_time, 100)],

            :z => [(:x, :prod_r, 0.001), (:z, :adhesion, 100), (:z, :volume, 50, 50), (:z, :perimeter, 2, 85)]
    )
))


open("elaborat/graph.dot", "w") do file
    write(file, output)
  end
