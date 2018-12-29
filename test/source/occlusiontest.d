module occlusiontest;

import tested;
import randommdp;
import discretemdp;
import discretefunctions;
import occlusion;
import std.typecons;
import std.conv;
import std.random;
import std.math;
import gridworld;

import std.stdio;


@name("Deterministic trajectory occlusion simple test")
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

    Tuple!(State) [] occ = new Tuple!(State)[1];
    occ[0] = states.toArray()[1];
    Set!State occluded_states = new Set!(State)(occ);
    Set!State[] occluded_states_arr = new Set!State[3];
    occluded_states_arr[] = occluded_states;
    
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

    ExactOccludedTrajectoryToTrajectoryDistr trajectoryCalc = new ExactOccludedTrajectoryToTrajectoryDistr(model, reward_obj, occluded_states_arr );

    auto distr = trajectoryCalc.to_traj_distr(arr, true_weights);


    assert(distr[0][0][0][tuple(states.toArray()[0][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in first entry " ~ to!string(distr));
    assert(distr[0][1][0][tuple(states.toArray()[1][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in unknown entry");
    assert(distr[0][2][0][tuple(states.toArray()[2][0], actions.toArray()[0][0])] == 1, "Trajectory calc failed in third entry");
    
}


@name("Random MDP occlusion test")
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

    Tuple!(State) [] occ = new Tuple!(State)[1];
    occ[0] = states.toArray()[1];
    Set!State occluded_states = new Set!(State)(occ);
    
 //   writeln(model.getOptimumPolicy());
    auto traj = new Sequence!(State, Action)(3);

    traj[0] = tuple(states.toArray()[0][0], actions.toArray()[1][0]);
    traj[1] = Tuple!(State, Action)(null, null);
    traj[2] = tuple(states.toArray()[2][0], actions.toArray()[0][0]);


    Sequence!(State, Action)[] arr = new Sequence!(State, Action)[1];
    arr[0] = traj;
    
    // the missing entry is obviously 1 => 1

    auto trajectoryCalc = new ExactStaticOccludedTrajectoryToTrajectoryDistr(model, reward_obj, occluded_states, 4 );
    auto controlCalc = new MarkovSmootherExactStaticOccludedTrajectoryToTrajectoryDistr(model, reward_obj, occluded_states, 4 );

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

        occluded_states = randomOccludedStates(model, uniform(1, 5));
        
        trajectoryCalc = new ExactStaticOccludedTrajectoryToTrajectoryDistr(model, lr, occluded_states, 15 );
        controlCalc = new MarkovSmootherExactStaticOccludedTrajectoryToTrajectoryDistr(model, lr, occluded_states, 15 );

        // generate random trajectories with random missing timesteps
        arr[0] = simulate(model, model.getPolicy(), uniform(10, 15), model.initialStateDistribution());

        arr[0] = removeOccludedTimesteps(arr[0], occluded_states);        

//        writeln("traj ", arr[0]);
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


@name("Gridworld MDP occlusion test")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.1;

    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {
        foreach (a ; actions) {

            auto newState = (cast(GridWorldAction)a[0]).getIdealStateFor(cast(GridWorldState)s[0]);

            Distribution!State ds = new Distribution!(State)(states, 0.0);

            if (states.contains(cast(State)newState)) {
                ds[newState] = 1.0;
            } else {
                ds[s[0]] = 1.0;
            }

            ds.normalize();
            
            transitions[s[0], a[0]] = ds;
        }
    }

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())) );

    // random occlusion
    foreach (i; 0 .. 10) {

        auto occluded_states = randomOccludedStates(model, uniform(cast(int)(0.1 * (sizeX * sizeY)), cast(int)(0.8 * (sizeX * sizeY))));
        
        auto trajectoryCalc = new ExactStaticOccludedTrajectoryToTrajectoryDistr(model, lr, occluded_states, 20 );
        auto controlCalc = new MarkovSmootherExactStaticOccludedTrajectoryToTrajectoryDistr(model, lr, occluded_states, 20 );

        // generate random trajectories with random missing timesteps
        auto arr = new Sequence!(State, Action)[1];
        arr[0] = simulate(model, model.getPolicy(), uniform(15, 20), model.initialStateDistribution());

        arr[0] = removeOccludedTimesteps(arr[0], occluded_states);        

//        writeln("traj ", arr[0]);
        auto distr = trajectoryCalc.to_traj_distr(arr, lr.getWeights());
        auto controlDistr = controlCalc.to_traj_distr(arr, lr.getWeights());

        foreach (s; model.S()) {
            foreach (a; model.A()) {
                foreach(t; 0 .. arr[0].length) {
                    assert(approxEqual(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])]), "4 - Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
                }
            }
        }
    }

    // chunks of the gridworld occluded
    foreach (i; 0 .. 16) {

        Tuple!(State) [] occ = new Tuple!(State)[(sizeX * sizeY) / 4];

        size_t count = 0;
        foreach (x ; 0 .. sizeX / 2) {
            foreach(y ; 0 .. sizeY / 2) {
                occ[count] = tuple(new GridWorldState((i % 2) * (sizeX / 2) + x, (i / 2) % 2 * (sizeY / 2) + y));
                count ++;
            }
        }
        
        auto occluded_states = new Set!State(occ);
        
        auto trajectoryCalc = new ExactStaticOccludedTrajectoryToTrajectoryDistr(model, lr, occluded_states, 20 );
        auto controlCalc = new MarkovSmootherExactStaticOccludedTrajectoryToTrajectoryDistr(model, lr, occluded_states, 20 );

        // generate random trajectories with random missing timesteps
        auto arr = new Sequence!(State, Action)[1];
        arr[0] = simulate(model, model.getPolicy(), uniform(15, 20), model.initialStateDistribution());

        arr[0] = removeOccludedTimesteps(arr[0], occluded_states);        

//        writeln("traj ", arr[0]);
        auto distr = trajectoryCalc.to_traj_distr(arr, lr.getWeights());
        auto controlDistr = controlCalc.to_traj_distr(arr, lr.getWeights());

        foreach (s; model.S()) {
            foreach (a; model.A()) {
                foreach(t; 0 .. arr[0].length) {
                    assert(approxEqual(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])]), "5 - Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
                }
            }
        }
    }    
}
