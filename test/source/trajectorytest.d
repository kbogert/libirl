module trajectorytest;

import tested;
import randommdp;
import discretemdp;
import discretefunctions;
import std.typecons;
import trajectories;
import std.math;
import std.conv;
import std.random;

import std.stdio;

@name("Deterministic Trajectory Missing Data test")
unittest {


    RandomMDPStateSpace states = new RandomMDPStateSpace(3);
    RandomMDPActionSpace actions = new RandomMDPActionSpace(2);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));
    foreach (s ; states) {
        foreach (a ; actions) {
            transitions[s[0], a[0]] = new Distribution!(State)(states, 0.0);

            auto rS = cast(RandomMDPState)s[0];
            auto rA = cast(RandomMDPAction)a[0];
            
            if (rA.getID() == 0) {
                // stay in current state
                transitions[tuple(s[0], a[0])][s[0]] = 1.0;
            } else {
                // move up one
                foreach (sp ; states) {
                    auto rSp = cast(RandomMDPState)sp[0];

                    if ((rSp.getID() == rS.getID() + 1) ||
                        (rS.getID() == states.size() - 1 && rSp.getID() == 0)) {
                            
                        transitions[tuple(s[0], a[0])][sp[0]] = 1.0;
                        break;
                    }
                }
            } 
        }
    }

    double [] true_weights = new double[3 * 2];
    foreach (ref w ; true_weights) {
        w = 0;
    }
    true_weights[$-1] = 1;
   
    auto reward_obj = new UniqueFeaturesPerStateActionReward(states, actions, true_weights);

    auto model = new BasicModel(states, actions, transitions, reward_obj.toFunction(), 0.95, new Distribution!(State)(states, DistInitType.Uniform), 0.1);

    auto traj = new Sequence!(State, Action)(3);

    traj[0] = tuple(states.toArray()[0][0], actions.toArray()[1][0]);
    traj[1] = Tuple!(State, Action)(null, null);
    traj[2] = tuple(states.toArray()[2][0], actions.toArray()[0][0]);


    Sequence!(State, Action)[] arr = new Sequence!(State, Action)[1];
    arr[0] = traj;
    
    // the missing entry is obviously 1 => 1

    ExactPartialTrajectoryToTrajectoryDistr trajectoryCalc = new ExactPartialTrajectoryToTrajectoryDistr(model, reward_obj );

    auto distr = trajectoryCalc.to_traj_distr(arr, true_weights);


    assert(distr[0][0][0][tuple(states.toArray()[0][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in first entry " ~ to!string(distr));
    assert(distr[0][1][0][tuple(states.toArray()[1][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in unknown entry");
    assert(distr[0][2][0][tuple(states.toArray()[2][0], actions.toArray()[0][0])] == 1, "Trajectory calc failed in third entry");
    

}

@name("Markov Smoother test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);
    
    auto results = SequenceMarkovChainSmoother!(size_t)(observations, transitions, initial);

    assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1");
    assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2");
    assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3");    
    assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4");
    assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5");

    assert(results[0][0].isNormalized(), "Not Normalized");
    assert(results[1][0].isNormalized(), "Not Normalized");
    assert(results[2][0].isNormalized(), "Not Normalized");
    assert(results[3][0].isNormalized(), "Not Normalized");
    assert(results[4][0].isNormalized(), "Not Normalized");
     
}


@name("Stochastic Trajectory Missing Data test")
unittest {


    RandomMDPStateSpace states = new RandomMDPStateSpace(3);
    RandomMDPActionSpace actions = new RandomMDPActionSpace(2);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));
    foreach (s ; states) {
        foreach (a ; actions) {
            transitions[s[0], a[0]] = new Distribution!(State)(states, 0.0);

            auto rS = cast(RandomMDPState)s[0];
            auto rA = cast(RandomMDPAction)a[0];
            
            
            if (rA.getID() == 0) {
                transitions[tuple(s[0], a[0])][s[0]] = 0.75;
                // barely move backward
                foreach (sp ; states) {
                    auto rSp = cast(RandomMDPState)sp[0];

                    if ((rSp.getID() == rS.getID() - 1) ||
                        (rSp.getID() == states.size() - 1 && rS.getID() == 0)) {
                            
                        transitions[tuple(s[0], a[0])][sp[0]] = 0.25;
                        break;
                    }
                }
            } else {
                transitions[tuple(s[0], a[0])][s[0]] = 0.25;
                // move up one
                foreach (sp ; states) {
                    auto rSp = cast(RandomMDPState)sp[0];

                    if ((rSp.getID() == rS.getID() + 1) ||
                        (rS.getID() == states.size() - 1 && rSp.getID() == 0)) {
                            
                        transitions[tuple(s[0], a[0])][sp[0]] = 0.750;
                        break;
                    }
                }
            } 
        }
    }

    double [] true_weights = new double[3 * 2];
    foreach (ref w ; true_weights) {
        w = 0;
    }
    true_weights[$-1] = 1;
   
    auto reward_obj = new UniqueFeaturesPerStateActionReward(states, actions, true_weights);

    auto model = new BasicModel(states, actions, transitions, reward_obj.toFunction(), 0.95, new Distribution!(State)(states, DistInitType.Uniform), 0.1);

 //   writeln(model.getOptimumPolicy());
    auto traj = new Sequence!(State, Action)(3);

    traj[0] = tuple(states.toArray()[0][0], actions.toArray()[1][0]);
    traj[1] = Tuple!(State, Action)(null, null);
    traj[2] = tuple(states.toArray()[2][0], actions.toArray()[0][0]);


    Sequence!(State, Action)[] arr = new Sequence!(State, Action)[1];
    arr[0] = traj;
    
    // the missing entry is obviously 1 => 1

    auto trajectoryCalc = new ExactPartialTrajectoryToTrajectoryDistr(model, reward_obj );
    auto controlCalc = new MarkovSmootherExactPartialTrajectoryToTrajectoryDistr(model, reward_obj );

    auto distr = trajectoryCalc.to_traj_distr(arr, true_weights);
    auto controlDistr = controlCalc.to_traj_distr(arr, true_weights);

    foreach (s; model.S()) {
        foreach (a; model.A()) {
            foreach(t; 0 .. arr[0].length) {

                assert(approxEqual(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])]), "Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
            }
        }
    }

    assert(distr[0][0][0][tuple(states.toArray()[0][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in first entry, " ~ to!string(distr));
    assert(distr[0][1][0][tuple(states.toArray()[1][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in unknown entry");
    assert(distr[0][2][0][tuple(states.toArray()[2][0], actions.toArray()[0][0])] == 1, "Trajectory calc failed in third entry");
    

    traj = new Sequence!(State, Action)(4);

    traj[0] = tuple(states.toArray()[0][0], actions.toArray()[1][0]);
    traj[1] = Tuple!(State, Action)(null, null);
    traj[2] = Tuple!(State, Action)(null, null);
    traj[3] = tuple(states.toArray()[2][0], actions.toArray()[0][0]);

    arr[0] = traj;

    distr = trajectoryCalc.to_traj_distr(arr, true_weights);
    controlDistr = controlCalc.to_traj_distr(arr, true_weights);
    
    foreach (s; model.S()) {
        foreach (a; model.A()) {
            foreach(t; 0 .. arr[0].length) {

                assert(approxEqual(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])]), "2 - Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
            }
        }
    }

    foreach (i; 0 .. 50) {

        UniqueFeaturesPerStateActionReward lr;
        model = generateRandomMDP(5, 3, 10, 1, 0.95, lr);

        trajectoryCalc = new ExactPartialTrajectoryToTrajectoryDistr(model, lr );
        controlCalc = new MarkovSmootherExactPartialTrajectoryToTrajectoryDistr(model, lr );

                
        // generate random trajectories with random missing timesteps
        arr[0] = simulate(model, model.getPolicy(), uniform(10, 15), model.initialStateDistribution());

        foreach(j; 0 .. arr[0].length) {
            if (uniform01() < 0.15) {
                arr[0][j] = Tuple!(State, Action)(null, null);
            }
        }        

        distr = trajectoryCalc.to_traj_distr(arr, lr.getWeights());
        controlDistr = controlCalc.to_traj_distr(arr, lr.getWeights());

        foreach (s; model.S()) {
            foreach (a; model.A()) {
                foreach(t; 0 .. arr[0].length) {

                    assert(approxEqual(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])]), "3 - Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
                }
            }
        }
    }
    
}


@name("Markov Gibbs Sampler test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);

    foreach (i; 0 .. 10) {
        auto results = MarkovGibbsSampler!(size_t)(observations, transitions, initial, 10000, 1000000);

        assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1 " ~ to!string(results[0][0][tuple(1UL)]));
        assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2 " ~ to!string(results[1][0][tuple(1UL)]));
        assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3 " ~ to!string(results[2][0][tuple(1UL)]));    
        assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4 " ~ to!string(results[3][0][tuple(1UL)]));
        assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5 " ~ to!string(results[4][0][tuple(1UL)]));

        assert(results[0][0].isNormalized(), "Not Normalized");
        assert(results[1][0].isNormalized(), "Not Normalized");
        assert(results[2][0].isNormalized(), "Not Normalized");
        assert(results[3][0].isNormalized(), "Not Normalized");
        assert(results[4][0].isNormalized(), "Not Normalized");
    }     
}

