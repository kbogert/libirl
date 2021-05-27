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
    traj[2] = tuple(states.toArray()[2][0], actions.toArray()[1][0]);


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

                assert(isClose(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])], 0.01, 1e-5), "Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
            }
        }
    }

    assert(distr[0][0][0][tuple(states.toArray()[0][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in first entry, " ~ to!string(distr));
    assert(distr[0][1][0][tuple(states.toArray()[1][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in unknown entry");
    assert(distr[0][2][0][tuple(states.toArray()[2][0], actions.toArray()[1][0])] == 1, "Trajectory calc failed in third entry");
    

    traj = new Sequence!(State, Action)(4);

    traj[0] = tuple(states.toArray()[0][0], actions.toArray()[1][0]);
    traj[1] = Tuple!(State, Action)(null, null);
    traj[2] = Tuple!(State, Action)(null, null);
    traj[3] = tuple(states.toArray()[2][0], actions.toArray()[1][0]);

    arr[0] = traj;

    distr = trajectoryCalc.to_traj_distr(arr, true_weights);
    controlDistr = controlCalc.to_traj_distr(arr, true_weights);
    
    foreach (s; model.S()) {
        foreach (a; model.A()) {
            foreach(t; 0 .. arr[0].length) {

                assert(isClose(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])], 0.01, 1e-5), "2 - Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
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

                    assert(isClose(distr[0][t][0][tuple(s[0], a[0])], controlDistr[0][t][0][tuple(s[0], a[0])], 0.01, 1e-5), "3 - Trajectory calc failed to match control, calc: " ~ to!string(distr) ~ " control: " ~ to!string(controlDistr)); 
                }
            }
        }
    }
    
}


