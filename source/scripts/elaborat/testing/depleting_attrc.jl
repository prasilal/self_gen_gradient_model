cfg = Dict(
    :size => (200, 200),
    :torus => (false,false),
    :seed => 1234,

    :sim => nothing,

    :vaxes_rule_steps => 10,

    :vaxes => @d(
        :v => @d(:D => 0,
                  :d => 0.0001,
                  :rd => 0.00001
                 ),
        :a => @d(:D => 0.5,
                  :d => 0.001,
                  :rd => 0.00001,
                  :show => true,
                  :init => (190:199, 1:199, 10)
                 ),
        :va => @d(:D => 0,
                  :d => 0.1, #0.001
                  :rd => 0.0001)

    ),

    :reactions => [
        @d(
            :react => [(1, :v), (1, :a)],
            :prod => [(1, :va)],
            :k => 0.1,
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
      @d(
        :receptors => @d(:v => 0.09),
        :state => @d(:cum_state => :v, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (35,40)
      ),

      @d(
        :receptors => @d(:v => 0.09),
        :state => @d(:cum_state => :v, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (30,44)
      ),

      @d(
        :receptors => @d(:v => 0.09),
        :state => @d(:cum_state => :v, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (40,36)
      ),

      @d(
        :receptors => @d(:v => 0.09),
        :state => @d(:cum_state => :v, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (45,32)
      )
      #=,

      @d(
        :receptors => @d(:va => 0.05),
        :state => @d(:cum_state => :a, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (50,50)
      )
      =#
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :v => [(:v, :prod_r, 0.01), (:v, :adhesion, 100), (:v, :volume, 50, 50), 
                   (:v, :perimeter, 2, 45)],
            :a => [(:a, :prod_v, 10000), (:a, :adhesion, 100), (:a, :volume, 50, 50), (:a, :perimeter, 2, 45)],
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

simulate(sim_desc, num_of_steps = 100)



output = string((print_zg(
  (
    :v => [(:v, :prod_r, 0.01), (:v, :adhesion, 100), (:v, :volume, 50, 50), 
           (:v, :perimeter, 2, 45)],
    :a => [(:a, :prod_v, 1000), (:a, :adhesion, 100), (:a, :volume, 50, 50), (:a, :perimeter, 2, 45)],
    :va => []
)
)))

open("graph.dot", "w") do file
  write(file, output)
end
