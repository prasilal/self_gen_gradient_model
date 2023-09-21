cfg = Dict(
    :size => (200, 200),
    :torus => (true, true),
    :seed => 1233,

    :sim => nothing,

    :vaxes_rule_steps => 40,

    :vaxes => @d(
        :v => @d(:D => 0,
                  :d => 0,
                  :rd => 0
                 ),
        :t => @d(:D => 0,
                 :d => 0,
                 :rd => 0
                ),
        :a => @d(:D => 0.5,
                  :d => 0,
                  :rd => 0,
                  :show => true,
                  :init => (95:105, 95:105, 10)
                 ),
        :x => @d(:D => 0,
                 :d => 0,
                 :rd => 0
                 ),
        :y => @d(:D => 0,
                 :d => 0,
                 :rd => 0
                 )

    ),

    :reactions => [
        @d(
            :react => [(1, :v), (1, :a)],
            :prod => [(1, :v)],
            :k => 0.01,
            :w => 1.0,
            :r_absorb => true,
        ),
        @d(
            :react => [(1, :t), (1, :a)],
            :prod => [(1, :t)],
            :k => 0.01,
            :w => 1.0,
            :r_absorb => false,
        ),
        @d(
            :react => [(1, :y), (1, :v)],
            :prod => [(1, :v)],
            :k => 0.01,
            :w => 10.0,
            :r_absorb => false,
        ),
        @d(
            :react => [(1, :x), (1, :t)],
            :prod => [(1, :t)],
            :k => 0.01,
            :w => 10.0,
            :r_absorb => false,
        )
    ],

    :cells => [
      @d(
        :receptors => @d(:v => 0.5),
        :state => @d(:cum_state => :v, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (20, 20)
      ),
      @d(
        :receptors => @d(:t => 0),
        :state => @d(:cum_state => :t, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (180, 180)
      ),
      @d(
        :receptors => @d(:y => 0.05, :x => 0.05),
        :state => @d(:cum_state => :a, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (100,100)
      )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :v => [(:v, :adhesion, 100), (:v, :volume, 50, 50), 
                   (:v, :perimeter, 2,60), (:a, :move, 22000)],
            :t => [(:t, :adhesion, 100), (:t, :volume, 50, 50), 
                   (:t, :perimeter, 2, 60), (:a, :move, 22000)],
            :a => [(:a, :prod_v, 500), (:a, :adhesion, 100), (:a, :volume, 50, 50), (:a, :perimeter, 2, 45)],
            :x =>[(:x, :adhesion, 100), (:x, :volume, 50, 50), (:x, :perimeter, 2,60)],
            :y =>[(:y, :adhesion, 100), (:y, :volume, 50, 50), (:y, :perimeter, 2,60)]
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    ),

    :runtime => @d()
)

sim_desc = init_sim(cfg)
sim_desc[:sim].vaxes[1][:,:] .= 1

simulate(sim_desc, num_of_steps = 1000)



output=(print_zg(
    (
        :v => [(:v, :adhesion, 100), (:v, :volume, 50, 50), 
                   (:v, :perimeter, 2,60), (:a, :move, 22000)],
            :t => [(:t, :adhesion, 100), (:t, :volume, 50, 50), 
                   (:t, :perimeter, 2, 60), (:a, :move, 22000)],
            :a => [(:a, :prod_v, 1000), (:a, :adhesion, 100), (:a, :volume, 50, 50), (:a, :perimeter, 2, 45)],
            :x =>[(:x, :adhesion, 100), (:x, :volume, 50, 50), (:x, :perimeter, 2,60)],
            :y =>[(:y, :adhesion, 100), (:y, :volume, 50, 50), (:y, :perimeter, 2,60)]
    )
))

open("elaborat/graph.dot", "w") do file
    write(file, output)
  end


